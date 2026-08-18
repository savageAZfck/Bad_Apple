//! Fail-closed, user-space filesystem automation.
//!
//! Actions are accepted only from `badapple-action` fenced JSON, validated
//! against canonical allowlisted roots, and executed directly with `std::fs`.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

pub const MAX_COPY_BYTES: u64 = 100 * 1024 * 1024;
const FENCE_OPEN: &str = "```badapple-action";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    CreateDirectory {
        path: PathBuf,
    },
    CopyFile {
        source: PathBuf,
        destination: PathBuf,
    },
    MoveFile {
        source: PathBuf,
        destination: PathBuf,
    },
    MoveToTrash {
        path: PathBuf,
    },
}

impl Action {
    pub fn operation(&self) -> &'static str {
        match self {
            Self::CreateDirectory { .. } => "create_directory",
            Self::CopyFile { .. } => "copy_file",
            Self::MoveFile { .. } => "move_file",
            Self::MoveToTrash { .. } => "move_to_trash",
        }
    }

    fn paths(&self) -> Vec<&Path> {
        match self {
            Self::CreateDirectory { path } | Self::MoveToTrash { path } => vec![path],
            Self::CopyFile {
                source,
                destination,
            }
            | Self::MoveFile {
                source,
                destination,
            } => {
                vec![source, destination]
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub action: Action,
    pub resolved_paths: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trash_destination: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExecutionReport {
    pub operation: String,
    pub paths: Vec<PathBuf>,
    pub result: String,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug)]
pub struct AutomationCage {
    roots: Vec<PathBuf>,
    home: PathBuf,
    log_path: PathBuf,
}

/// Parse every exact `badapple-action` fenced block, in source order.
/// Each block must contain exactly one JSON action object.
pub fn parse_actions(input: &str) -> Result<Vec<Action>> {
    let normalized = input.replace("\r\n", "\n");
    let mut actions = Vec::new();
    let mut lines = normalized.lines();
    while let Some(line) = lines.next() {
        if line.trim() != FENCE_OPEN {
            continue;
        }
        let mut json = String::new();
        let mut closed = false;
        for body_line in lines.by_ref() {
            if body_line.trim() == "```" {
                closed = true;
                break;
            }
            if !json.is_empty() {
                json.push('\n');
            }
            json.push_str(body_line);
        }
        if !closed {
            bail!("unterminated badapple-action fence");
        }
        if json.trim().is_empty() {
            bail!("empty badapple-action fence");
        }
        actions.push(serde_json::from_str(&json).context("invalid badapple-action JSON")?);
    }
    if actions.is_empty() {
        bail!("no badapple-action fenced JSON found");
    }
    Ok(actions)
}

impl AutomationCage {
    pub fn from_env() -> Result<Self> {
        let home = home_directory()?;
        let roots = match env::var_os("BADAPPLE_AUTOMATION_ROOTS") {
            Some(raw) => {
                let parsed: Vec<_> = env::split_paths(&raw).collect();
                if parsed.is_empty() {
                    bail!("BADAPPLE_AUTOMATION_ROOTS contains no roots");
                }
                parsed
            }
            None => default_roots(&home)?,
        };
        let log_path = env::var_os("BADAPPLE_AUTOMATION_LOG")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".badapple").join("automation.jsonl"));
        Self::new_with_log(roots, home, log_path)
    }

    pub fn new(roots: Vec<PathBuf>) -> Result<Self> {
        let home = home_directory()?;
        let log_path = home.join(".badapple").join("automation.jsonl");
        Self::new_with_log(roots, home, log_path)
    }

    pub fn new_with_log(roots: Vec<PathBuf>, home: PathBuf, log_path: PathBuf) -> Result<Self> {
        if roots.is_empty() {
            bail!("automation requires at least one allowlisted root");
        }
        let mut canonical_roots = Vec::new();
        for root in roots {
            reject_lexical_path(&root)?;
            let canonical = fs::canonicalize(&root)
                .with_context(|| format!("allowlisted root does not exist: {}", root.display()))?;
            if !canonical.is_dir() {
                bail!("allowlisted root is not a directory: {}", root.display());
            }
            reject_forbidden_path(&canonical, &home)?;
            if !canonical_roots.contains(&canonical) {
                canonical_roots.push(canonical);
            }
        }
        Ok(Self {
            roots: canonical_roots,
            home,
            log_path,
        })
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    pub fn validate(&self, action: &Action) -> Result<Plan> {
        let mut resolved_paths = Vec::new();
        for path in action.paths() {
            resolved_paths.push(self.resolve_allowlisted(path)?);
        }
        let trash_destination = match action {
            Action::CreateDirectory { .. } => {
                if resolved_paths[0].exists() {
                    bail!(
                        "destination already exists: {}",
                        resolved_paths[0].display()
                    );
                }
                None
            }
            Action::CopyFile { .. } => {
                validate_regular_source(&resolved_paths[0])?;
                if fs::metadata(&resolved_paths[0])?.len() > MAX_COPY_BYTES {
                    bail!("copy source exceeds the {MAX_COPY_BYTES}-byte limit");
                }
                validate_destination(&resolved_paths[0], &resolved_paths[1])?;
                None
            }
            Action::MoveFile { .. } => {
                validate_regular_source(&resolved_paths[0])?;
                validate_destination(&resolved_paths[0], &resolved_paths[1])?;
                None
            }
            Action::MoveToTrash { .. } => {
                validate_trash_source(&resolved_paths[0])?;
                Some(self.next_trash_destination(&resolved_paths[0])?)
            }
        };
        Ok(Plan {
            action: action.clone(),
            resolved_paths,
            trash_destination,
        })
    }

    /// Revalidates immediately before mutation and writes a content-free JSONL audit record.
    pub fn execute(&self, action: &Action) -> Result<ExecutionReport> {
        let started = Instant::now();
        let operation = action.operation().to_owned();
        let paths: Vec<PathBuf> = action.paths().into_iter().map(Path::to_path_buf).collect();
        let result = self.execute_inner(action);
        let report = ExecutionReport {
            operation,
            paths,
            result: match &result {
                Ok(()) => "success".to_owned(),
                Err(error) => format!("error: {error:#}"),
            },
            elapsed_ms: started.elapsed().as_millis(),
        };
        self.append_log(&report)?;
        result?;
        Ok(report)
    }

    fn execute_inner(&self, action: &Action) -> Result<()> {
        let plan = self.validate(action)?;
        match action {
            Action::CreateDirectory { .. } => {
                create_directory_chain(&plan.resolved_paths[0], &self.roots)
            }
            Action::CopyFile { .. } => {
                fs::copy(&plan.resolved_paths[0], &plan.resolved_paths[1])
                    .context("copy_file failed")?;
                Ok(())
            }
            Action::MoveFile { .. } => {
                fs::rename(&plan.resolved_paths[0], &plan.resolved_paths[1])
                    .context("move_file failed")?;
                Ok(())
            }
            Action::MoveToTrash { .. } => {
                let destination = plan
                    .trash_destination
                    .ok_or_else(|| anyhow!("missing trash destination"))?;
                fs::rename(&plan.resolved_paths[0], destination).context("move_to_trash failed")?;
                Ok(())
            }
        }
    }

    fn resolve_allowlisted(&self, path: &Path) -> Result<PathBuf> {
        reject_lexical_path(path)?;
        reject_forbidden_path(path, &self.home)?;
        let resolved = canonicalize_existing_ancestor(path)?;
        reject_forbidden_path(&resolved, &self.home)?;
        if !self.roots.iter().any(|root| resolved.starts_with(root)) {
            bail!("path escapes all allowlisted roots: {}", path.display());
        }
        Ok(resolved)
    }

    fn next_trash_destination(&self, source: &Path) -> Result<PathBuf> {
        let trash = self.home.join(".Trash");
        let canonical_trash = fs::canonicalize(&trash)
            .with_context(|| format!("trash directory is unavailable: {}", trash.display()))?;
        if !canonical_trash.is_dir() || !canonical_trash.starts_with(&self.home) {
            bail!("unsafe trash directory");
        }
        let name = source
            .file_name()
            .ok_or_else(|| anyhow!("trash source has no file name"))?;
        let first = canonical_trash.join(name);
        if !first.exists() {
            return Ok(first);
        }
        let (stem, extension) = split_name(name);
        for suffix in 1_u32..=10_000 {
            let candidate_name = match &extension {
                Some(extension) => format!("{stem} {suffix}.{extension}"),
                None => format!("{stem} {suffix}"),
            };
            let candidate = canonical_trash.join(candidate_name);
            if !candidate.exists() {
                return Ok(candidate);
            }
        }
        bail!("unable to choose a collision-safe trash name")
    }

    fn append_log(&self, report: &ExecutionReport) -> Result<()> {
        let parent = self
            .log_path
            .parent()
            .ok_or_else(|| anyhow!("automation log path has no parent"))?;
        fs::create_dir_all(parent).context("unable to create automation log directory")?;
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .context("unable to open automation audit log")?;
        serde_json::to_writer(&mut log, report).context("unable to encode automation audit log")?;
        log.write_all(b"\n")?;
        log.flush()?;
        Ok(())
    }
}

fn home_directory() -> Result<PathBuf> {
    env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set"))
}

fn default_roots(home: &Path) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    for name in [
        "Desktop",
        "Documents",
        "Downloads",
        "Developer",
        "Projects",
        "src",
    ] {
        let candidate = home.join(name);
        if candidate.is_dir() {
            roots.push(candidate);
        }
    }
    let cwd = env::current_dir()?;
    if cwd.starts_with(home) && (cwd.join(".git").exists() || cwd.join("Cargo.toml").is_file()) {
        roots.push(cwd);
    }
    if roots.is_empty() {
        bail!("no safe default automation roots exist; set BADAPPLE_AUTOMATION_ROOTS");
    }
    Ok(roots)
}

fn reject_lexical_path(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("automation paths must be absolute: {}", path.display());
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir | Component::CurDir) {
            bail!(
                "path traversal components are forbidden: {}",
                path.display()
            );
        }
        if let Component::Normal(name) = component {
            reject_security_name(name, path)?;
        }
    }
    Ok(())
}

fn reject_security_name(name: &OsStr, path: &Path) -> Result<()> {
    let lowered = name.to_string_lossy().to_ascii_lowercase();
    let forbidden = [
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".git",
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "authorized_keys",
        "known_hosts",
    ];
    if forbidden.contains(&lowered.as_str())
        || lowered.starts_with(".env.")
        || lowered.ends_with(".keychain-db")
    {
        bail!("hidden security path is forbidden: {}", path.display());
    }
    Ok(())
}

fn reject_forbidden_path(path: &Path, home: &Path) -> Result<()> {
    let forbidden = [
        Path::new("/"),
        Path::new("/System"),
        Path::new("/private"),
        Path::new("/Library"),
        Path::new("/root"),
        Path::new("/etc"),
        Path::new("/usr"),
        Path::new("/bin"),
        Path::new("/sbin"),
        Path::new("/var"),
    ];
    if forbidden
        .iter()
        .any(|prefix| path == *prefix || path.starts_with(prefix))
        || path == home.join("Library")
        || path.starts_with(home.join("Library"))
    {
        bail!("system or private path is forbidden: {}", path.display());
    }
    Ok(())
}

fn canonicalize_existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut ancestor = path;
    let mut missing = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .ok_or_else(|| anyhow!("path has no existing ancestor: {}", path.display()))?;
        missing.push(name.to_owned());
        ancestor = ancestor
            .parent()
            .ok_or_else(|| anyhow!("path has no existing ancestor: {}", path.display()))?;
    }
    let metadata = fs::symlink_metadata(ancestor)?;
    if metadata.file_type().is_symlink() {
        bail!("symlink path components are forbidden: {}", path.display());
    }
    let mut resolved = fs::canonicalize(ancestor)?;
    for name in missing.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn validate_regular_source(source: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("source does not exist: {}", source.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!(
            "source must be a regular, non-symlink file: {}",
            source.display()
        );
    }
    Ok(())
}

fn validate_trash_source(source: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("trash source does not exist: {}", source.display()))?;
    let kind = metadata.file_type();
    if kind.is_symlink() || !(kind.is_file() || kind.is_dir()) {
        bail!("trash source must be a regular file or directory");
    }
    Ok(())
}

fn validate_destination(source: &Path, destination: &Path) -> Result<()> {
    if source == destination {
        bail!("source and destination must differ");
    }
    if destination.exists() {
        bail!(
            "destination already exists; overwrite is forbidden: {}",
            destination.display()
        );
    }
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("destination has no parent"))?;
    let metadata = fs::symlink_metadata(parent)
        .with_context(|| format!("destination parent does not exist: {}", parent.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("destination parent must be an existing non-symlink directory");
    }
    Ok(())
}

fn create_directory_chain(path: &Path, roots: &[PathBuf]) -> Result<()> {
    let root = roots
        .iter()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.components().count())
        .ok_or_else(|| anyhow!("directory is outside allowlisted roots"))?;
    let relative = path.strip_prefix(root)?;
    let mut current = root.clone();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            bail!("invalid directory component");
        };
        current.push(name);
        if current.exists() {
            let metadata = fs::symlink_metadata(&current)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("unsafe existing directory component: {}", current.display());
            }
        } else {
            fs::create_dir(&current)
                .with_context(|| format!("unable to create directory: {}", current.display()))?;
        }
    }
    Ok(())
}

fn split_name(name: &OsStr) -> (String, Option<String>) {
    let name = name.to_string_lossy();
    match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() && !extension.is_empty() => {
            (stem.to_owned(), Some(extension.to_owned()))
        }
        _ => (name.into_owned(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        cage: AutomationCage,
    }
    impl Fixture {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let base = env::temp_dir().join(format!("badapple-cage-{}-{id}", std::process::id()));
            let root = base.join("allowed");
            let home = base.join("home");
            fs::create_dir_all(home.join(".Trash")).unwrap();
            fs::create_dir_all(&root).unwrap();
            let cage =
                AutomationCage::new_with_log(vec![root.clone()], home, base.join("audit.jsonl"))
                    .unwrap();
            Self { base, root, cage }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    #[test]
    fn parses_fenced_action_and_rejects_unknown_fields_and_operations() {
        let parsed = parse_actions("text\n```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/a\"}\n```").unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parse_actions("```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/a\",\"mode\":511}\n```").is_err());
        assert!(parse_actions(
            "```badapple-action\n{\"operation\":\"delete_file\",\"path\":\"/tmp/a\"}\n```"
        )
        .is_err());
    }

    #[test]
    fn rejects_traversal_root_and_hidden_security_paths() {
        let fixture = Fixture::new();
        for path in [
            fixture.root.join("../escape"),
            PathBuf::from("/"),
            PathBuf::from("/System/x"),
            fixture.root.join(".ssh/id_ed25519"),
        ] {
            assert!(fixture
                .cage
                .validate(&Action::CreateDirectory { path })
                .is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let outside = fixture.base.join("outside");
        fs::create_dir(&outside).unwrap();
        symlink(&outside, fixture.root.join("link")).unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CreateDirectory {
                path: fixture.root.join("link/escaped")
            })
            .is_err());
    }

    #[test]
    fn executes_allowed_create_copy_move_and_trash() {
        let fixture = Fixture::new();
        let directory = fixture.root.join("nested/dir");
        fixture
            .cage
            .execute(&Action::CreateDirectory {
                path: directory.clone(),
            })
            .unwrap();
        let source = directory.join("source.txt");
        fs::write(&source, b"safe test data").unwrap();
        let copy = directory.join("copy.txt");
        fixture
            .cage
            .execute(&Action::CopyFile {
                source: source.clone(),
                destination: copy.clone(),
            })
            .unwrap();
        let moved = directory.join("moved.txt");
        fixture
            .cage
            .execute(&Action::MoveFile {
                source: copy,
                destination: moved.clone(),
            })
            .unwrap();
        fixture
            .cage
            .execute(&Action::MoveToTrash { path: moved })
            .unwrap();
        assert!(fixture.cage.home.join(".Trash/moved.txt").exists());
        assert!(fixture.base.join("audit.jsonl").exists());
    }

    #[test]
    fn rejects_same_destination_overwrite_directory_and_oversized_copy() {
        let fixture = Fixture::new();
        let source = fixture.root.join("source");
        fs::write(&source, b"x").unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: source.clone(),
                destination: source.clone()
            })
            .is_err());
        let destination = fixture.root.join("destination");
        fs::write(&destination, b"existing").unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: source.clone(),
                destination
            })
            .is_err());
        let directory_source = fixture.root.join("directory");
        fs::create_dir(&directory_source).unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: directory_source,
                destination: fixture.root.join("copy")
            })
            .is_err());
        let oversized = fixture.root.join("oversized");
        let file = fs::File::create(&oversized).unwrap();
        file.set_len(MAX_COPY_BYTES + 1).unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: oversized,
                destination: fixture.root.join("large-copy")
            })
            .is_err());
    }
}

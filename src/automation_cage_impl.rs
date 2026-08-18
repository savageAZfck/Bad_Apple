//! Fail-closed local filesystem automation cage.
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

pub const MAX_COPY_BYTES: u64 = 100 * 1024 * 1024;

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

/// Parse every `badapple-action` or `json` fenced block that deserializes
/// into a valid Action. This is a fail-closed fallback for models that may
/// label the code fence as `json` while still emitting a valid action object.
pub fn parse_actions(input: &str) -> Result<Vec<Action>> {
    let input = input.replace("\r\n", "\n");
    let mut lines = input.lines().peekable();
    let mut out = Vec::new();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        let is_open = trimmed == "```badapple-action" || trimmed == "```json";
        if !is_open {
            continue;
        }
        let mut body = String::new();
        let mut closed = false;
        for line in lines.by_ref() {
            if line.trim() == "```" {
                closed = true;
                break;
            }
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(line);
        }
        if !closed || body.trim().is_empty() {
            bail!("invalid or unterminated action fence");
        }
        if let Ok(action) = serde_json::from_str::<Action>(&body) {
            out.push(action);
        }
    }
    if out.is_empty() {
        bail!("no badapple-action or compatible json fenced action found");
    }
    Ok(out)
}

impl AutomationCage {
    pub fn from_env() -> Result<Self> {
        let home = home_directory()?;
        let roots = if let Some(raw) = env::var_os("BADAPPLE_AUTOMATION_ROOTS") {
            let roots: Vec<_> = env::split_paths(&raw).collect();
            if roots.is_empty() {
                bail!("BADAPPLE_AUTOMATION_ROOTS is empty");
            }
            roots
        } else {
            default_roots(&home)?
        };
        let log_path = env::var_os("BADAPPLE_AUTOMATION_LOG")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".badapple/automation.jsonl"));
        Self::new_with_log(roots, home, log_path)
    }

    pub fn new(roots: Vec<PathBuf>) -> Result<Self> {
        let home = home_directory()?;
        Self::new_with_log(roots, home.clone(), home.join(".badapple/automation.jsonl"))
    }

    pub fn new_with_log(roots: Vec<PathBuf>, home: PathBuf, log_path: PathBuf) -> Result<Self> {
        if roots.is_empty() {
            bail!("at least one allowlisted root is required");
        }
        let mut safe_roots = Vec::new();
        for root in roots {
            reject_lexical_path(&root)?;
            let root = fs::canonicalize(&root).context("allowlisted root does not exist")?;
            reject_forbidden_path(&root, &home)?;
            if !root.is_dir() {
                bail!("allowlisted root is not a directory");
            }
            if !safe_roots.contains(&root) {
                safe_roots.push(root);
            }
        }
        Ok(Self {
            roots: safe_roots,
            home,
            log_path,
        })
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    pub fn validate(&self, action: &Action) -> Result<Plan> {
        let paths: Vec<_> = action
            .paths()
            .into_iter()
            .map(|path| self.resolve_allowlisted(path))
            .collect::<Result<_>>()?;
        let trash_destination = match action {
            Action::CreateDirectory { .. } => {
                if paths[0].exists() {
                    bail!("destination exists; overwrite forbidden");
                }
                None
            }
            Action::CopyFile { .. } => {
                validate_regular_source(&paths[0])?;
                if fs::metadata(&paths[0])?.len() > MAX_COPY_BYTES {
                    bail!("copy source is oversized");
                }
                validate_destination(&paths[0], &paths[1])?;
                None
            }
            Action::MoveFile { .. } => {
                validate_regular_source(&paths[0])?;
                validate_destination(&paths[0], &paths[1])?;
                None
            }
            Action::MoveToTrash { .. } => {
                let metadata =
                    fs::symlink_metadata(&paths[0]).context("trash source does not exist")?;
                if metadata.file_type().is_symlink() || !(metadata.is_file() || metadata.is_dir()) {
                    bail!("special or symlink trash source forbidden");
                }
                Some(self.next_trash_destination(&paths[0])?)
            }
        };
        Ok(Plan {
            action: action.clone(),
            resolved_paths: paths,
            trash_destination,
        })
    }

    /// Revalidates before mutation and appends a content-free offline JSONL record.
    pub fn execute(&self, action: &Action) -> Result<ExecutionReport> {
        let started = Instant::now();
        let result = self.execute_inner(action);
        let report = ExecutionReport {
            operation: action.operation().into(),
            paths: action.paths().into_iter().map(Path::to_path_buf).collect(),
            result: result
                .as_ref()
                .map(|_| "success".into())
                .unwrap_or_else(|error| format!("error: {error:#}")),
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
                    .context("copy failed")?;
                Ok(())
            }
            Action::MoveFile { .. } => {
                fs::rename(&plan.resolved_paths[0], &plan.resolved_paths[1]).context("move failed")
            }
            Action::MoveToTrash { .. } => fs::rename(
                &plan.resolved_paths[0],
                plan.trash_destination
                    .ok_or_else(|| anyhow!("missing trash destination"))?,
            )
            .context("trash move failed"),
        }
    }

    fn resolve_allowlisted(&self, path: &Path) -> Result<PathBuf> {
        reject_lexical_path(path)?;
        reject_forbidden_path(path, &self.home)?;
        let resolved = canonicalize_existing_ancestor(path)?;
        reject_forbidden_path(&resolved, &self.home)?;
        if !self.roots.iter().any(|root| resolved.starts_with(root)) {
            bail!("path escapes allowlisted roots");
        }
        Ok(resolved)
    }

    fn next_trash_destination(&self, source: &Path) -> Result<PathBuf> {
        let trash = fs::canonicalize(self.home.join(".Trash")).context("~/.Trash unavailable")?;
        if !trash.starts_with(&self.home) || !trash.is_dir() {
            bail!("unsafe trash directory");
        }
        let name = source
            .file_name()
            .ok_or_else(|| anyhow!("source has no name"))?
            .to_string_lossy();
        for suffix in 0..=10_000 {
            let candidate = if suffix == 0 {
                trash.join(name.as_ref())
            } else {
                trash.join(format!("{name} {suffix}"))
            };
            if !candidate.exists() {
                return Ok(candidate);
            }
        }
        bail!("no collision-safe trash name available")
    }

    fn append_log(&self, report: &ExecutionReport) -> Result<()> {
        let parent = self
            .log_path
            .parent()
            .ok_or_else(|| anyhow!("invalid log path"))?;
        fs::create_dir_all(parent)?;
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        serde_json::to_writer(&mut log, report)?;
        log.write_all(b"\n")?;
        log.flush()?;
        Ok(())
    }
}

fn home_directory() -> Result<PathBuf> {
    env::var_os("HOME")
        .filter(|value| !value.is_empty())
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
        let path = home.join(name);
        if path.is_dir() {
            roots.push(path);
        }
    }
    if let Ok(entries) = fs::read_dir(home) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let source_root = path.is_dir()
                && [
                    ".git",
                    "Cargo.toml",
                    "Package.swift",
                    "pyproject.toml",
                    "go.mod",
                    "package.json",
                ]
                .iter()
                .any(|marker| path.join(marker).exists());
            if source_root {
                roots.push(path);
            }
        }
    }
    let cwd = env::current_dir()?;
    if cwd.starts_with(home) && (cwd.join(".git").exists() || cwd.join("Cargo.toml").is_file()) {
        roots.push(cwd);
    }
    roots.sort();
    roots.dedup();
    if roots.is_empty() {
        bail!("no safe default roots; set BADAPPLE_AUTOMATION_ROOTS");
    }
    Ok(roots)
}

fn reject_lexical_path(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("paths must be absolute");
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir | Component::CurDir) {
            bail!("traversal forbidden");
        }
        if let Component::Normal(name) = component {
            let name = name.to_string_lossy().to_ascii_lowercase();
            if [
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
            ]
            .contains(&name.as_str())
                || name.starts_with(".env.")
                || name.ends_with(".keychain-db")
            {
                bail!("hidden security path forbidden");
            }
        }
    }
    Ok(())
}

fn reject_forbidden_path(path: &Path, home: &Path) -> Result<()> {
    let forbidden = [
        "/System", "/private", "/Library", "/root", "/etc", "/usr", "/bin", "/sbin", "/var",
    ];
    if path == Path::new("/")
        || forbidden.iter().any(|prefix| path.starts_with(prefix))
        || path.starts_with(home.join("Library"))
    {
        bail!("root, system, private, or Library path forbidden");
    }
    Ok(())
}

fn canonicalize_existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut ancestor = path;
    let mut missing = Vec::new();
    while !ancestor.exists() {
        missing.push(
            ancestor
                .file_name()
                .ok_or_else(|| anyhow!("no existing ancestor"))?
                .to_owned(),
        );
        ancestor = ancestor
            .parent()
            .ok_or_else(|| anyhow!("no existing ancestor"))?;
    }
    if fs::symlink_metadata(ancestor)?.file_type().is_symlink() {
        bail!("symlink path forbidden");
    }
    let mut resolved = fs::canonicalize(ancestor)?;
    for name in missing.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn validate_regular_source(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).context("source does not exist")?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("source must be a regular non-symlink file");
    }
    Ok(())
}

fn validate_destination(source: &Path, destination: &Path) -> Result<()> {
    if source == destination {
        bail!("source equals destination");
    }
    if destination.exists() {
        bail!("overwrite forbidden");
    }
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("destination has no parent"))?;
    let metadata = fs::symlink_metadata(parent).context("destination parent missing")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("unsafe destination parent");
    }
    Ok(())
}

fn create_directory_chain(path: &Path, roots: &[PathBuf]) -> Result<()> {
    let root = roots
        .iter()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.components().count())
        .ok_or_else(|| anyhow!("outside roots"))?;
    let mut current = root.clone();
    for component in path.strip_prefix(root)?.components() {
        let Component::Normal(name) = component else {
            bail!("invalid component");
        };
        current.push(name);
        if current.exists() {
            let metadata = fs::symlink_metadata(&current)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("unsafe directory component");
            }
        } else {
            fs::create_dir(&current)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        cage: AutomationCage,
    }

    impl Fixture {
        fn new() -> Self {
            let base = home_directory().unwrap().join(".badapple").join(format!(
                "automation-cage-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let root = base.join("allowed");
            let fixture_home = base.join("home");
            fs::create_dir_all(fixture_home.join(".Trash")).unwrap();
            fs::create_dir_all(&root).unwrap();
            let cage = AutomationCage::new_with_log(
                vec![root.clone()],
                fixture_home,
                base.join("audit.jsonl"),
            )
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
    fn parsing_accepts_json_fence_and_other_valid_fences() {
        assert_eq!(
            parse_actions("```json\n{\"operation\":\"create_directory\",\"path\":\"/tmp/x\"}\n```")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            parse_actions(
                "```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/x\"}\n```\n```json\n{\"operation\":\"move_to_trash\",\"path\":\"/tmp/y\"}\n```"
            )
            .unwrap()
            .len(),
            2
        );
        assert!(parse_actions("```json\n{\"not\":\"an action\"}\n```").is_err());
    }

    #[test]
    fn parsing_rejects_unknown_fields_and_operations() {
        assert_eq!(
            parse_actions(
                "```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/x\"}\n```"
            )
            .unwrap()
            .len(),
            1
        );
        assert!(parse_actions(
            "```badapple-action\n{\"operation\":\"delete_file\",\"path\":\"/tmp/x\"}\n```"
        )
        .is_err());
        assert!(parse_actions("```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/x\",\"mode\":1}\n```").is_err());
    }

    #[test]
    fn rejects_traversal_root_and_hidden_security_paths() {
        let fixture = Fixture::new();
        for path in [
            fixture.root.join("../escape"),
            PathBuf::from("/"),
            PathBuf::from("/System/x"),
            fixture.root.join(".ssh/key"),
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
        symlink(outside, fixture.root.join("link")).unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CreateDirectory {
                path: fixture.root.join("link/escape"),
            })
            .is_err());
    }

    #[test]
    fn allowed_temp_root_actions_and_rejections() {
        let fixture = Fixture::new();
        let directory = fixture.root.join("nested/dir");
        fixture
            .cage
            .execute(&Action::CreateDirectory {
                path: directory.clone(),
            })
            .unwrap();
        let source = directory.join("source");
        fs::write(&source, b"safe data").unwrap();
        let copy = directory.join("copy");
        fixture
            .cage
            .execute(&Action::CopyFile {
                source: source.clone(),
                destination: copy.clone(),
            })
            .unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: source.clone(),
                destination: source.clone(),
            })
            .is_err());
        let moved = directory.join("moved");
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
        assert!(fixture.cage.home.join(".Trash/moved").exists());

        let oversized = directory.join("oversized");
        let file = fs::File::create(&oversized).unwrap();
        file.set_len(MAX_COPY_BYTES + 1).unwrap();
        assert!(fixture
            .cage
            .validate(&Action::CopyFile {
                source: oversized,
                destination: directory.join("oversized-copy"),
            })
            .is_err());
    }
}

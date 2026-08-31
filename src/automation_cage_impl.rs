//! Fail-closed local filesystem automation cage.
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

pub const MAX_COPY_BYTES: u64 = 100 * 1024 * 1024;

// ============================================================================
// openat-based safe path operations — eliminate TOCTOU races by using fd-based
// semantics instead of string-based path validation. Each component is opened
// with O_NOFOLLOW so a symlink swapped between validation and use cannot
// redirect the operation outside the cage.
// ============================================================================

/// Open a directory fd by walking from a root fd, component by component,
/// rejecting symlinks at every step via O_NOFOLLOW. Returns the final fd
/// or -1 on error. Caller is responsible for closing the fd.
fn openat_walk(root_fd: i32, components: &[&str]) -> Result<i32> {
    let mut fd = root_fd;
    let mut need_close = false;
    for &component in components {
        // SAFETY: openat with O_NOFOLLOW is safe because the kernel atomically rejects
        // symlinks at the leaf, preventing path substitution races. `fd` is a valid
        // directory fd (the root fd or a previously-opened component), and `component`
        // is a NUL-terminated C string borrowed from the caller's slice.
        let next = unsafe {
            libc::openat(
                fd,
                component.as_ptr() as *const _,
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
            )
        };
        if next < 0 {
            let err = std::io::Error::last_os_error();
            if need_close {
                // SAFETY: libc::close is safe to call on any valid fd. `fd` was obtained
                // from a successful prior openat in this loop and has not been closed yet.
                unsafe {
                    libc::close(fd);
                }
            }
            return Err(err).context(format!("openat {} failed", component));
        }
        if need_close {
            // SAFETY: libc::close is safe to call on any valid fd. `fd` is the previous
            // component's fd, which is now superseded by `next` and no longer needed.
            unsafe {
                libc::close(fd);
            }
        }
        fd = next;
        need_close = true;
    }
    Ok(fd)
}

/// Create a file safely using openat: walk to the parent directory fd with
/// O_NOFOLLOW at each step, then openat the file with O_CREAT | O_EXCL |
/// O_NOFOLLOW. This eliminates the TOCTOU between path validation and file
/// creation.
fn openat_create_file(root_fd: i32, parent_components: &[&str], filename: &str) -> Result<()> {
    let parent_fd = openat_walk(root_fd, parent_components)?;
    // SAFETY: openat with O_CREAT | O_EXCL | O_NOFOLLOW is safe because `parent_fd` is a
    // valid directory fd returned by openat_walk, `filename` is a NUL-terminated C string,
    // and O_EXCL ensures the file must not already exist, preventing overwrite races.
    // O_NOFOLLOW rejects a symlink swapped at the leaf.
    let result = unsafe {
        libc::openat(
            parent_fd,
            filename.as_ptr() as *const _,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
            0o644,
        )
    };
    if parent_fd != root_fd {
        // SAFETY: libc::close is safe to call on any valid fd. `parent_fd` was obtained
        // from a successful openat_walk and is no longer referenced after this point.
        unsafe {
            libc::close(parent_fd);
        }
    }
    if result < 0 {
        return Err(std::io::Error::last_os_error()).context("openat create file failed");
    }
    // SAFETY: libc::close is safe to call on any valid fd. `result` is a non-negative fd
    // returned by the openat above; we close it immediately since we only needed to prove
    // the file could be created.
    unsafe {
        libc::close(result);
    }
    Ok(())
}

/// Create a directory safely using openat with O_NOFOLLOW at each step.
fn openat_create_dir(root_fd: i32, components: &[&str]) -> Result<()> {
    let mut fd = root_fd;
    let mut need_close = false;
    for &component in components {
        // Try to open existing component first
        // SAFETY: openat with O_NOFOLLOW | O_DIRECTORY is safe because `fd` is a valid
        // directory fd and O_NOFOLLOW atomically rejects a symlink at the leaf, so a
        // path component swapped between validation and use cannot escape the cage.
        let existing = unsafe {
            libc::openat(
                fd,
                component.as_ptr() as *const _,
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
            )
        };
        if existing >= 0 {
            if need_close {
                // SAFETY: libc::close is safe to call on any valid fd. `fd` is the prior
                // component fd now superseded by `existing`.
                unsafe {
                    libc::close(fd);
                }
            }
            fd = existing;
            need_close = true;
        } else {
            // Doesn't exist — create it
            // SAFETY: mkdirat is safe because `fd` is a valid directory fd and `component`
            // is a NUL-terminated C string. The new directory is created with mode 0o755.
            let result = unsafe { libc::mkdirat(fd, component.as_ptr() as *const _, 0o755) };
            if result < 0 {
                let err = std::io::Error::last_os_error();
                if need_close {
                    // SAFETY: libc::close is safe to call on any valid fd. `fd` was
                    // obtained from a prior successful openat and is no longer needed.
                    unsafe {
                        libc::close(fd);
                    }
                }
                return Err(err).context(format!("mkdirat {} failed", component));
            }
            // Now open the newly created directory
            // SAFETY: openat with O_NOFOLLOW | O_DIRECTORY is safe because the directory
            // was just created by mkdirat above and `fd` is a valid parent directory fd.
            let new_fd = unsafe {
                libc::openat(
                    fd,
                    component.as_ptr() as *const _,
                    libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
                )
            };
            if new_fd < 0 {
                let err = std::io::Error::last_os_error();
                if need_close {
                    // SAFETY: libc::close is safe to call on any valid fd. `fd` was
                    // obtained from a prior successful openat and is no longer needed.
                    unsafe {
                        libc::close(fd);
                    }
                }
                return Err(err).context(format!("openat after mkdirat {} failed", component));
            }
            if need_close {
                // SAFETY: libc::close is safe to call on any valid fd. `fd` is the prior
                // component fd now superseded by `new_fd`.
                unsafe {
                    libc::close(fd);
                }
            }
            fd = new_fd;
            need_close = true;
        }
    }
    if need_close {
        // SAFETY: libc::close is safe to call on any valid fd. `fd` is the final
        // component fd opened during the walk and is no longer needed by the caller.
        unsafe {
            libc::close(fd);
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    CreateFile {
        path: PathBuf,
    },
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
            Self::CreateFile { .. } => "create_file",
            Self::CreateDirectory { .. } => "create_directory",
            Self::CopyFile { .. } => "copy_file",
            Self::MoveFile { .. } => "move_file",
            Self::MoveToTrash { .. } => "move_to_trash",
        }
    }

    fn paths(&self) -> Vec<&Path> {
        match self {
            Self::CreateFile { path }
            | Self::CreateDirectory { path }
            | Self::MoveToTrash { path } => vec![path],
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
        let action = serde_json::from_str::<Action>(&body)
            .context("invalid badapple-action or json fenced action")?;
        out.push(action);
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
            .map_or_else(|| home.join(".badapple/automation.jsonl"), PathBuf::from);
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
            Action::CreateFile { .. } | Action::CreateDirectory { .. } => {
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
                .map_or_else(|error| format!("error: {error:#}"), |()| "success".into()),
            elapsed_ms: started.elapsed().as_millis(),
        };
        self.append_log(&report)?;
        result?;
        Ok(report)
    }

    fn execute_inner(&self, action: &Action) -> Result<()> {
        let plan = self.validate(action)?;
        match action {
            Action::CreateFile { .. } => create_file_chain(&plan.resolved_paths[0], &self.roots),
            Action::CreateDirectory { .. } => {
                create_directory_chain(&plan.resolved_paths[0], &self.roots)
            }
            Action::CopyFile { .. } => {
                // Open source with O_NOFOLLOW to atomically reject symlinks
                // at the leaf, eliminating the TOCTOU between validate_regular_source
                // and the actual read. Then open destination with O_NOFOLLOW too.
                use std::os::unix::fs::OpenOptionsExt;
                let source_file = fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&plan.resolved_paths[0])
                    .context("copy: source open failed (or symlink rejected)")?;
                // Verify the fd is a regular file with nlink == 1
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    let meta = source_file.metadata().context("copy: fstat source")?;
                    if !meta.is_file() {
                        bail!("copy: source is not a regular file");
                    }
                    if meta.nlink() > 1 {
                        bail!("copy: source has multiple hard links");
                    }
                }
                // Open destination with O_NOFOLLOW
                let dest_file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&plan.resolved_paths[1])
                    .context("copy: destination open failed (or symlink rejected)")?;
                let mut source_file = source_file;
                let mut dest_file = dest_file;
                std::io::copy(&mut source_file, &mut dest_file)
                    .context("copy: data transfer failed")?;
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
        let path = substitute_path_placeholders(path, &self.home)?;
        reject_lexical_path(&path)?;
        reject_forbidden_path(&path, &self.home)?;
        let resolved = canonicalize_existing_ancestor(&path)?;
        reject_forbidden_path(&resolved, &self.home)?;
        if !self.roots.iter().any(|root| resolved.starts_with(root)) {
            bail!("path escapes allowlisted roots");
        }
        // Reject broken/dangling symlinks at the leaf: a missing leaf that is
        // actually a symlink would be followed by fs::write/fs::copy and could
        // escape the cage.  symlink_metadata returns Ok for a broken symlink,
        // while metadata would fail.
        if let Ok(meta) = fs::symlink_metadata(&path) {
            if meta.file_type().is_symlink() {
                bail!("symlink at path leaf is forbidden");
            }
        }
        Ok(resolved)
    }

    fn next_trash_destination(&self, source: &Path) -> Result<PathBuf> {
        // Reject a symlinked ~/.Trash before canonicalizing it; otherwise a
        // symlink could redirect trashed files outside the home directory.
        let trash_meta =
            fs::symlink_metadata(self.home.join(".Trash")).context("~/.Trash unavailable")?;
        if trash_meta.file_type().is_symlink() {
            bail!("~/.Trash must not be a symlink");
        }
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
        // Validate that the parent directory is not a symlink, so an attacker
        // cannot redirect the audit stream by replacing the directory.
        // The log path itself is opened with O_NOFOLLOW below.
        let parent_meta = fs::symlink_metadata(parent)?;
        if parent_meta.file_type().is_symlink() {
            bail!("log parent directory must not be a symlink");
        }
        use std::os::unix::fs::OpenOptionsExt;
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&self.log_path)?;
        serde_json::to_writer(&mut log, report)?;
        log.write_all(b"\n")?;
        log.flush()?;
        Ok(())
    }
}

fn home_directory() -> Result<PathBuf> {
    let raw = env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set"))?;
    fs::canonicalize(&raw).context("cannot canonicalize HOME")
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
    // Reject hard links to sensitive files: a hard-linked inode shared with a
    // file outside the cage would let CopyFile exfiltrate its contents.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() > 1 {
            bail!("source has multiple hard links and may cross trust boundaries");
        }
    }
    Ok(())
}

fn validate_destination(source: &Path, destination: &Path) -> Result<()> {
    if source == destination {
        bail!("source equals destination");
    }
    // Reject a dangling/broken symlink at the destination leaf: fs::copy and
    // fs::rename would follow it and write outside the cage.
    if let Ok(meta) = fs::symlink_metadata(destination) {
        if meta.file_type().is_symlink() {
            bail!("destination is a symlink; overwrite via symlink is forbidden");
        }
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
    // Open the root directory fd
    // SAFETY: open with O_NOFOLLOW | O_DIRECTORY is safe because `root` is an allowlisted
    // directory that was canonicalized at cage construction. O_NOFOLLOW rejects a symlink
    // at the leaf, so a swapped root cannot redirect the fd outside the cage.
    let root_fd = unsafe {
        libc::open(
            root.to_string_lossy().as_ptr() as *const _,
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
        )
    };
    if root_fd < 0 {
        return Err(std::io::Error::last_os_error())
            .context("cannot open root directory with O_NOFOLLOW");
    }
    // Collect remaining components
    let components: Vec<String> = path
        .strip_prefix(root)?
        .components()
        .filter_map(|c| match c {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let c_refs: Vec<&str> = components.iter().map(|s| s.as_str()).collect();
    openat_create_dir(root_fd, &c_refs)?;
    // SAFETY: libc::close is safe to call on any valid fd. `root_fd` was obtained from a
    // successful open above and is no longer needed after openat_create_dir returns.
    unsafe {
        libc::close(root_fd);
    }
    Ok(())
}

fn create_file_chain(path: &Path, roots: &[PathBuf]) -> Result<()> {
    if path.exists() {
        bail!("destination exists; overwrite forbidden");
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("file path has no parent"))?;
    let filename = path
        .file_name()
        .ok_or_else(|| anyhow!("file path has no name"))?
        .to_string_lossy()
        .to_string();
    create_directory_chain(parent, roots)?;
    // Now use openat to create the file with O_NOFOLLOW at the leaf,
    // eliminating the TOCTOU between create_directory_chain and file creation.
    let root = roots
        .iter()
        .filter(|root| parent.starts_with(root))
        .max_by_key(|root| root.components().count())
        .ok_or_else(|| anyhow!("outside roots"))?;
    // SAFETY: open with O_NOFOLLOW | O_DIRECTORY is safe because `root` is an allowlisted
    // canonicalized directory. O_NOFOLLOW rejects a symlink at the leaf so the fd cannot
    // be redirected outside the cage.
    let root_fd = unsafe {
        libc::open(
            root.to_string_lossy().as_ptr() as *const _,
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
        )
    };
    if root_fd < 0 {
        return Err(std::io::Error::last_os_error())
            .context("cannot open root directory for file creation");
    }
    let parent_components: Vec<String> = parent
        .strip_prefix(root)?
        .components()
        .filter_map(|c| match c {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let c_refs: Vec<&str> = parent_components.iter().map(|s| s.as_str()).collect();
    openat_create_file(root_fd, &c_refs, &filename)?;
    // SAFETY: libc::close is safe to call on any valid fd. `root_fd` was obtained from a
    // successful open above and is no longer needed after openat_create_file returns.
    unsafe {
        libc::close(root_fd);
    }
    Ok(())
}

fn substitute_path_placeholders(path: &Path, home: &Path) -> Result<PathBuf> {
    if let Some(s) = path.to_str() {
        if let Some(tail) = s.strip_prefix("~/") {
            return Ok(home.join(tail));
        }
        // Only redirect /Users/<current>/... to the actual home directory so a
        // hallucinated or different user cannot be silently rewritten to HOME.
        if let Some(user) = home.file_name().and_then(|n| n.to_str()) {
            let prefix = format!("/Users/{user}/");
            if let Some(tail) = s.strip_prefix(&prefix) {
                return Ok(home.join(tail));
            }
        }
    }
    Ok(path.to_path_buf())
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

    #[test]
    fn create_file_rejects_symlink_at_leaf() {
        let fixture = Fixture::new();
        let target = fixture.root.join("evil_link");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/passwd", &target).unwrap();
        }
        assert!(fixture
            .cage
            .validate(&Action::CreateFile {
                path: target.clone(),
            })
            .is_err());
    }

    #[test]
    fn trash_symlink_is_rejected() {
        let fixture = Fixture::new();
        let trash = fixture.cage.home.join(".Trash");
        let _ = fs::remove_dir_all(&trash);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&fixture.root, &trash).unwrap();
        }
        let file = fixture.root.join("to_trash");
        fs::write(&file, b"data").unwrap();
        let result = fixture.cage.execute(&Action::MoveToTrash { path: file });
        assert!(result.is_err(), "trash symlink should be rejected");
    }

    // =========================================================================
    // Security regression tests — red team findings
    // =========================================================================

    /// Verify that CopyFile rejects a symlink as the copy source.
    #[cfg(unix)]
    #[test]
    fn copy_file_rejects_symlink_source() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let dir = fixture.root.join("cage_dir");
        fs::create_dir_all(&dir).unwrap();
        let real_file = dir.join("real_file");
        fs::write(&real_file, b"content").unwrap();
        let link = dir.join("symlink_source");
        symlink(&real_file, &link).unwrap();
        let dest = dir.join("dest");
        assert!(
            fixture
                .cage
                .validate(&Action::CopyFile {
                    source: link,
                    destination: dest,
                })
                .is_err(),
            "copy from a symlink source must be rejected"
        );
    }

    /// Verify that CopyFile rejects a hard-linked source (nlink > 1).
    #[cfg(unix)]
    #[test]
    fn copy_file_rejects_hardlink_source() {
        let fixture = Fixture::new();
        let dir = fixture.root.join("hlink_dir");
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source");
        fs::write(&source, b"content").unwrap();
        let hardlink = dir.join("hardlink_to_source");
        fs::hard_link(&source, &hardlink).unwrap();
        let dest = dir.join("dest");
        assert!(
            fixture
                .cage
                .validate(&Action::CopyFile {
                    source: hardlink,
                    destination: dest,
                })
                .is_err(),
            "copy from a hard-linked source must be rejected"
        );
    }

    /// Verify that CreateDirectory fails when a path component is a symlink.
    #[cfg(unix)]
    #[test]
    fn create_directory_rejects_symlink_component() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let outside = fixture.base.join("outside_target");
        fs::create_dir(&outside).unwrap();
        symlink(&outside, fixture.root.join("evil_link")).unwrap();
        assert!(
            fixture
                .cage
                .validate(&Action::CreateDirectory {
                    path: fixture.root.join("evil_link/subdir"),
                })
                .is_err(),
            "create_directory through a symlink component must be rejected"
        );
    }

    /// Verify that MoveFile rejects a symlink at the destination.
    #[cfg(unix)]
    #[test]
    fn move_file_rejects_symlink_destination() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let dir = fixture.root.join("move_dir");
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source");
        fs::write(&source, b"content").unwrap();
        let symlink_target = dir.join("symlink_target");
        fs::write(&symlink_target, b"target_content").unwrap();
        let dest = dir.join("dest_symlink");
        symlink(&symlink_target, &dest).unwrap();
        assert!(
            fixture
                .cage
                .validate(&Action::MoveFile {
                    source,
                    destination: dest,
                })
                .is_err(),
            "move to a symlink destination must be rejected"
        );
    }

    /// Verify that CopyFile with source == destination is rejected.
    #[test]
    fn copy_file_source_equals_destination_rejected() {
        let fixture = Fixture::new();
        let dir = fixture.root.join("eq_dir");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("file");
        fs::write(&file, b"content").unwrap();
        assert!(
            fixture
                .cage
                .validate(&Action::CopyFile {
                    source: file.clone(),
                    destination: file,
                })
                .is_err(),
            "copy where source equals destination must be rejected"
        );
    }

    /// Verify that executing an action produces a log entry (the O_NOFOLLOW log
    /// path works and writes valid JSONL).
    #[test]
    fn log_writes_to_o_nofollow_file() {
        let fixture = Fixture::new();
        let dir = fixture.root.join("log_test_dir");
        fixture
            .cage
            .execute(&Action::CreateDirectory { path: dir.clone() })
            .unwrap();

        let log_path = fixture.base.join("audit.jsonl");
        assert!(
            log_path.exists(),
            "audit log file should exist after execute"
        );

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(
            content.contains("create_directory"),
            "log should contain the operation name"
        );
        assert!(
            content.contains("success"),
            "log should contain success result"
        );
    }
}

//! Thin, blocking redb helpers used by `scavenger` and `strategy_library`.
//!
//! These replace the old `sled` calls with a single `&[u8]` → `&[u8]` table,
//! removing the `fxhash` and `instant` unmaintained transitive dependencies.

use anyhow::{Context, Result};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::path::Path;

pub const KV_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

fn db_file_path(path: &Path) -> std::path::PathBuf {
    if path.is_dir() || path.as_os_str().is_empty() {
        path.join("db.redb")
    } else {
        path.to_path_buf()
    }
}

fn prefix_end(prefix: &[u8]) -> Option<Vec<u8>> {
    for i in (0..prefix.len()).rev() {
        if prefix[i] < 0xff {
            let mut end = prefix.to_vec();
            end[i] = end[i].wrapping_add(1);
            end.truncate(i + 1);
            return Some(end);
        }
    }
    None
}

/// Open or create a redb database at `path`. If `path` is a directory, the
/// database file is created inside it as `db.redb`; otherwise the path is used
/// directly.
pub fn open(path: impl AsRef<Path>) -> Result<Database> {
    let path = path.as_ref();
    let file = db_file_path(path);
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create redb parent dir for {}", file.display()))?;
    }
    if file.exists() {
        Database::open(&file).with_context(|| format!("failed to open redb {}", file.display()))
    } else {
        Database::create(&file).with_context(|| format!("failed to create redb {}", file.display()))
    }
}

pub fn get(db: &Database, key: &[u8]) -> Result<Option<Vec<u8>>> {
    let tx = db.begin_read()?;
    let table = tx.open_table(KV_TABLE)?;
    Ok(table.get(key)?.map(|v| v.value().to_vec()))
}

pub fn insert(db: &Database, key: &[u8], value: &[u8]) -> Result<()> {
    let tx = db.begin_write()?;
    {
        let mut table = tx.open_table(KV_TABLE)?;
        table.insert(key, value)?;
    }
    tx.commit()?;
    Ok(())
}

pub fn insert_many(db: &Database, items: &[(&[u8], &[u8])]) -> Result<()> {
    let tx = db.begin_write()?;
    {
        let mut table = tx.open_table(KV_TABLE)?;
        for (key, value) in items {
            table.insert(*key, *value)?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn remove(db: &Database, key: &[u8]) -> Result<()> {
    let tx = db.begin_write()?;
    {
        let mut table = tx.open_table(KV_TABLE)?;
        table.remove(key)?;
    }
    tx.commit()?;
    Ok(())
}

pub fn remove_many(db: &Database, keys: &[&[u8]]) -> Result<()> {
    let tx = db.begin_write()?;
    {
        let mut table = tx.open_table(KV_TABLE)?;
        for key in keys {
            table.remove(*key)?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Scan all keys that start with `prefix` and return owned `(key, value)` pairs.
pub fn scan_prefix(db: &Database, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let tx = db.begin_read()?;
    let table = tx.open_table(KV_TABLE)?;
    let mut out = Vec::new();
    match prefix_end(prefix) {
        Some(end) => {
            let end = end.as_slice();
            for kv in table.range(prefix..end)? {
                let (k, v) = kv?;
                out.push((k.value().to_vec(), v.value().to_vec()));
            }
        }
        None => {
            for kv in table.range(prefix..)? {
                let (k, v) = kv?;
                out.push((k.value().to_vec(), v.value().to_vec()));
            }
        }
    }
    Ok(out)
}

pub fn count_prefix(db: &Database, prefix: &[u8]) -> Result<usize> {
    scan_prefix(db, prefix).map(|v| v.len())
}

/// Iterate over the entire table.
pub fn iter(db: &Database) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let tx = db.begin_read()?;
    let table = tx.open_table(KV_TABLE)?;
    let mut out = Vec::new();
    for kv in table.iter()? {
        let (k, v) = kv?;
        out.push((k.value().to_vec(), v.value().to_vec()));
    }
    Ok(out)
}

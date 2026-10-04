//! badapple-potlatch — the organism's wallet: compute IOUs between
//! sovereign organisms, settled against treaty invocations.
//!
//! Thin wrapper over the public `potlatch` crate.
//!
//! State under /var/lib/bad_apple/potlatch/:
//!   identity.key   — economic identity (ed25519, 0600)
//!   book.json      — the hash-chained ledger of instruments
//!   pending/       — issued IOUs awaiting debtor acceptance
//!
//!   badapple-potlatch init
//!   badapple-potlatch issue --debtor <pubkey> --treaty <id>
//!       --resource <kind> --amount <n> [--due <epoch>]
//!   badapple-potlatch accept <iou.json>          # sign as debtor
//!   badapple-potlatch invoke <iou-id> <units>    # log work performed
//!   badapple-potlatch settle <iou-id> <inv-id>[,<inv-id>...]
//!   badapple-potlatch default <iou-id> --reason <text>
//!   badapple-potlatch balance <iou-id>
//!   badapple-potlatch obligations [<pubkey>]     # debts owed
//!   badapple-potlatch defaults [<pubkey>]        # credit report
//!   badapple-potlatch book
//!   badapple-potlatch verify

use anyhow::{bail, Context, Result};
use potlatch::{Book, Identity, Iou};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/potlatch";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_POTLATCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        bail!("odd-length hex");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?))
        .collect()
}

fn load_identity() -> Result<Identity> {
    let path = root().join("identity.key");
    if path.exists() {
        let bytes: [u8; 32] = hex_decode(fs::read_to_string(&path)?.trim())
            .context("identity.key is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("identity.key must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(&root())?;
    let id = Identity::generate();
    fs::write(&path, hex_encode(id.seed()))?;
    Ok(id)
}

fn load_book() -> Result<Book> {
    let path = root().join("book.json");
    if path.exists() {
        let book: Book = serde_json::from_str(&fs::read_to_string(&path)?)?;
        book.verify()?;
        Ok(book)
    } else {
        Ok(Book::new())
    }
}

fn save_book(book: &Book) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(root().join("book.json"), serde_json::to_string_pretty(book)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-potlatch <init|issue|accept|invoke|settle|default|balance|obligations|defaults|book|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "init" => {
            let id = load_identity()?;
            fs::create_dir_all(root().join("pending"))?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "issue" => {
            let debtor = named("--debtor").context("--debtor required")?;
            let treaty = named("--treaty").context("--treaty required")?;
            let resource = named("--resource").context("--resource required")?;
            let amount: u64 = named("--amount")
                .context("--amount required")?
                .parse()?;
            let due = named("--due").and_then(|d| d.parse().ok());
            let id = load_identity()?;
            let mut book = load_book()?;
            let iou = book.issue(&id, &debtor, &treaty, &resource, amount, due);
            save_book(&book)?;
            let dir = root().join("pending");
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{}.json", &iou.id[..16]));
            fs::write(&path, serde_json::to_string_pretty(&iou)?)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "issued-pending-acceptance",
                    "iou": iou.id, "amount": amount,
                    "file": path,
                })
            );
        }
        "accept" => {
            let path = args.get(1).context("accept <iou.json>")?;
            let iou: Iou = serde_json::from_str(&fs::read_to_string(path)?)?;
            let id = load_identity()?;
            let mut book = load_book()?;
            let iou = book.accept(iou, &id)?;
            save_book(&book)?;
            println!("{}", serde_json::json!({"status":"accepted","iou":iou.id}));
        }
        "invoke" => {
            let iou_id = args.get(1).context("invoke <iou-id> <units>")?;
            let units: u64 = args.get(2).context("invoke <iou-id> <units>")?.parse()?;
            let id = load_identity()?;
            let mut book = load_book()?;
            let inv = book.invoke(&id, iou_id, units);
            save_book(&book)?;
            println!(
                "{}",
                serde_json::json!({"status":"invoked","invocation":inv.id,"units":units})
            );
        }
        "settle" => {
            let iou_id = args.get(1).context("settle <iou-id> <inv-ids>")?;
            let ids: Vec<&str> = args
                .get(2)
                .context("settle <iou-id> <inv-ids>")?
                .split(',')
                .collect();
            let id = load_identity()?;
            let mut book = load_book()?;
            let s = book.settle(&id, iou_id, &ids)?;
            save_book(&book)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "settled", "units": s.units,
                    "remaining": book.balance(iou_id)?,
                })
            );
        }
        "default" => {
            let iou_id = args.get(1).context("default <iou-id>")?;
            let reason = named("--reason").unwrap_or_else(|| "unsettled".into());
            let id = load_identity()?;
            let mut book = load_book()?;
            book.declare_default(&id, iou_id, &reason)?;
            save_book(&book)?;
            println!("{}", serde_json::json!({"status":"defaulted","iou":iou_id}));
        }
        "balance" => {
            let iou_id = args.get(1).context("balance <iou-id>")?;
            let book = load_book()?;
            println!(
                "{}",
                serde_json::json!({"iou":iou_id,"balance":book.balance(iou_id)?})
            );
        }
        "obligations" => {
            let id = load_identity()?;
            let who = args.get(1).cloned().unwrap_or_else(|| id.public_key());
            let book = load_book()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&book.obligations(&who))?
            );
        }
        "defaults" => {
            let id = load_identity()?;
            let who = args.get(1).cloned().unwrap_or_else(|| id.public_key());
            let book = load_book()?;
            println!("{}", serde_json::to_string_pretty(&book.defaults(&who))?);
        }
        "book" => {
            let book = load_book()?;
            println!(
                "{}",
                serde_json::json!({
                    "events": book.events.len(),
                    "ious": book.ious().len(),
                })
            );
        }
        "verify" => {
            let book = load_book()?;
            book.verify()?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","events":book.events.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}

use anyhow::Context;
use bad_apple::scavenger::{Scavenger, ScavengerConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = ScavengerConfig::from_env().context("scavenger config")?;
    eprintln!("scavenger watch dirs: {:?}", config.watch_dirs);

    let scavenger = Arc::new(Scavenger::open(config)?);
    scavenger.seed_existing_files().await?;

    Ok(())
}

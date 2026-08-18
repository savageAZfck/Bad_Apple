use anyhow::{bail, Context, Result};
use bad_apple::automation_cage::{parse_actions, AutomationCage};
use serde::Serialize;
use std::io::{self, Read};

const MAX_INPUT_BYTES: u64 = 1024 * 1024;

#[derive(Serialize)]
struct PlanOutput<'a, T> {
    mode: &'a str,
    plans: T,
}

fn main() -> Result<()> {
    let mut confirmed = false;
    let mut input_parts = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--confirm" if !confirmed => confirmed = true,
            "--confirm" => bail!("--confirm may be supplied only once"),
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "--" => {
                input_parts.extend(args);
                break;
            }
            _ if arg.starts_with('-') => bail!("unknown option: {arg}"),
            _ => input_parts.push(arg),
        }
    }

    let input = if input_parts.is_empty() {
        let mut bytes = Vec::new();
        io::stdin()
            .take(MAX_INPUT_BYTES + 1)
            .read_to_end(&mut bytes)
            .context("unable to read automation input")?;
        if bytes.len() as u64 > MAX_INPUT_BYTES {
            bail!("automation input exceeds the {MAX_INPUT_BYTES}-byte limit");
        }
        String::from_utf8(bytes).context("automation input must be UTF-8")?
    } else {
        input_parts.join(" ")
    };

    let actions = parse_actions(&input)?;
    let cage = AutomationCage::from_env()?;
    // Validate the complete batch before allowing its first mutation.
    let plans = actions
        .iter()
        .map(|action| cage.validate(action))
        .collect::<Result<Vec<_>>>()?;

    if !confirmed {
        println!(
            "{}",
            serde_json::to_string_pretty(&PlanOutput {
                mode: "validate_only",
                plans
            })?
        );
        return Ok(());
    }

    let reports = actions
        .iter()
        .map(|action| cage.execute(action))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&PlanOutput {
            mode: "executed",
            plans: reports
        })?
    );
    Ok(())
}

fn print_help() {
    println!(
        "badapple_automation — fail-closed local filesystem automation\n\n\
         Usage:\n  badapple_automation [--confirm] [FENCED_ACTION_TEXT]\n\n\
         Reads UTF-8 input from stdin when no text argument is supplied. Without\n\
         --confirm, actions are validated and a plan is printed but no mutation occurs.\n\n\
         Environment:\n  BADAPPLE_AUTOMATION_ROOTS  Colon-separated allowlisted roots\n  \
         BADAPPLE_AUTOMATION_LOG    JSONL audit log path"
    );
}

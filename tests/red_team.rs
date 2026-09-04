//! Red-team harness integration tests.

use bad_apple::red_team::{RedTeamRunner, Severity};

#[test]
fn red_team_default_registry_passes_all_probes() {
    let runner = RedTeamRunner::default();
    let report = runner.run_once();

    eprintln!(
        "[redteam] total: {}, mitigated: {}, score: {}",
        report.total, report.mitigated, report.score
    );

    assert!(
        !report.attempts.is_empty(),
        "red-team runner should execute at least one probe"
    );
    assert!(
        report.score >= 0.99,
        "red-team score should be at least 0.99, got {} with {} findings",
        report.score,
        report.findings().len()
    );
}

#[test]
fn red_team_cage_category_passes() {
    let runner = RedTeamRunner::default();
    let report = runner.run_category("cage");
    assert!(report.total > 0, "cage category should have probes");
    assert!(
        report.findings().is_empty(),
        "cage probes produced findings: {:?}",
        report.findings()
    );
}

#[test]
fn red_team_slicks_category_passes() {
    let runner = RedTeamRunner::default();
    let report = runner.run_category("slicks");
    assert!(report.total > 0, "slicks category should have probes");
    assert!(
        report.findings().is_empty(),
        "slicks probes produced findings: {:?}",
        report.findings()
    );
}

#[test]
fn red_team_p2p_category_passes() {
    let runner = RedTeamRunner::default();
    let report = runner.run_category("p2p");
    assert!(report.total > 0, "p2p category should have probes");
    assert!(
        report.findings().is_empty(),
        "p2p probes produced findings: {:?}",
        report.findings()
    );
}

#[test]
fn red_team_wasm_category_passes() {
    let runner = RedTeamRunner::default();
    let report = runner.run_category("wasm");
    assert!(report.total > 0, "wasm category should have probes");
    assert!(
        report.findings().is_empty(),
        "wasm probes produced findings: {:?}",
        report.findings()
    );
}

#[test]
fn red_team_finding_serialization_round_trip() {
    let finding = bad_apple::red_team::Finding {
        probe: "test".to_string(),
        category: "test".to_string(),
        severity: Severity::High,
        title: "test".to_string(),
        description: "test".to_string(),
        payload: Some("payload".to_string()),
        timestamp_ms: 0,
        duration_us: 0,
    };
    let json = serde_json::to_string(&finding).expect("serializes");
    let _: bad_apple::red_team::Finding = serde_json::from_str(&json).expect("deserializes");
}

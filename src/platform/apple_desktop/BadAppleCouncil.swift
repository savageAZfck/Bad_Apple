// BadAppleCouncil — the co-council of decision makers.
//
// Fourteen deterministic seats — the four financial minds of MOG-ARET
// (Buffett, Dalio, Musk, Jobs) plus ten strategists (Sun Tzu, Clausewitz,
// Musashi, Machiavelli, Napoleon, Hannibal, Aurelius, Boyd, Genghis, Patton) —
// vote on every gated action before it runs.
//
// A proposed tool call is encoded into a fixed feature vector; each seat
// applies its own personality transform over those features and returns a
// vote in [-1, 1] with a one-line rationale. Weighted consensus produces the
// verdict plus a dissent score. The whole deliberation is deterministic and
// journaled — the record shows not just what was decided but what each mind
// voted and why.
//
// Authority model: in manual mode the verdict rides on approval prompts as
// counsel to the user. Under autopilot the council holds the gate — majority
// approve with dissent under the ceiling executes; deny or contested votes
// escalate back to a human proposal. IFY remains the posterior brake; the
// council is the prior deliberation layer.

import Foundation

// MARK: - Action features

/// A proposed action reduced to a fixed numeric vector the seats vote on.
/// Every field is normalized to 0...1.
struct ActionFeatures {
    /// Does the action destroy, overwrite, send, or otherwise remove state?
    var destructiveness: Double = 0.0
    /// 1.0 = cannot be undone, 0.0 = fully reversible.
    var irreversibility: Double = 0.0
    /// Reach of effect: 0 = single temp file, 1 = system-wide or external network.
    var blastRadius: Double = 0.0
    /// Touches credentials, keys, personal data, or protected paths.
    var targetSensitivity: Double = 0.0
    /// Requires elevated privilege (sudo, launchd, admin APIs).
    var privilege: Double = 0.0
    /// Breadth of objects affected (wildcards, recursion, whole directories).
    var scopeMass: Double = 0.0
    /// Resource or time cost of the action.
    var cost: Double = 0.0
    /// How unusual this action is versus routine use (0 = routine).
    var novelty: Double = 0.3

    /// Aggregate risk — the shared baseline every seat weighs differently.
    var risk: Double {
        0.30 * destructiveness
            + 0.22 * irreversibility
            + 0.16 * blastRadius
            + 0.12 * targetSensitivity
            + 0.10 * privilege
            + 0.05 * scopeMass
            + 0.05 * cost
    }
}

/// Heuristic encoder: tool name + arguments → ActionFeatures.
enum ActionEncoder {

    static func encode(toolName: String, args: [String: String]) -> ActionFeatures {
        var f = ActionFeatures()
        let text = args.values.joined(separator: " ").lowercased()
        let path = (args["path"] ?? args["file"] ?? args["dir"] ?? "").lowercased()
        let cmd = (args["command"] ?? args["script"] ?? "").lowercased()

        // Shared text signals.
        let destructiveTerms = ["rm ", "rm -", "delete", "remove", "unlink", "rmdir",
                                "dd ", "mkfs", "format", "truncate", "> /dev", "shred",
                                "drop ", "overwrite", "kill ", "pkill", "killall",
                                "shutdown", "reboot", "halt", "poweroff", "kill -9",
                                "kill -s", "pmset sleep", "umount", "wipedisk"]
        let privilegeTerms = ["sudo", "do shell script", "launchctl", "dscl", "chmod 777",
                              "chown", "csrutil", "nvram", "systemsetup", "diskutil"]
        let sensitiveTerms = [".ssh", ".aws", ".gnupg", "keychain", "id_rsa", "id_ed25519",
                              "password", "secret", "token", "credential", ".env",
                              "/etc/", "/var/", "/system/", "/usr/", "/bin/", "library/keychains"]
        let networkTerms = ["curl", "wget", "http://", "https://", "ssh ", "scp ", "ftp",
                            "nc ", "ncat", "socket", "send", "mail", "smtp"]
        let massTerms = ["-r", "-rf", "-f", "*", "...", "--recursive", "all"]

        func termScore(_ terms: [String], _ haystack: String) -> Double {
            let hits = terms.filter { haystack.contains($0) }.count
            return min(1.0, Double(hits) * 0.34)
        }

        // Piped remote execution ("curl x.sh | sh", "wget -O- u | bash") is
        // unverifiable code from the network — near-worst-case by construction.
        let pipedShell = cmd.contains("| sh") || cmd.contains("|sh")
            || cmd.contains("| bash") || cmd.contains("|bash")
            || cmd.contains("| zsh") || cmd.contains("sh -c") && cmd.contains("curl")
            || cmd.contains("bash -c") && cmd.contains("curl")

        switch toolName {
        case "run_shell":
            f.destructiveness = termScore(destructiveTerms, " " + cmd)
            if cmd.contains("rm") && (cmd.contains("-r") || cmd.contains("-f")) {
                f.destructiveness = max(f.destructiveness, 0.85)
            }
            if pipedShell {
                f.destructiveness = max(f.destructiveness, 0.75)
                f.blastRadius = max(f.blastRadius, 0.8)
            }
            f.irreversibility = f.destructiveness > 0.3 ? 0.9 : 0.35
            if cmd.contains("git checkout") || cmd.contains("git restore") { f.irreversibility = 0.75 }
            f.blastRadius = cmd.contains("sudo") || cmd.hasPrefix("launchctl") ? 0.7 : 0.35
            f.privilege = termScore(privilegeTerms, cmd)
            f.scopeMass = termScore(massTerms, cmd)
            f.targetSensitivity = termScore(sensitiveTerms, cmd + " " + text)
            f.cost = cmd.contains("curl") || cmd.contains("wget") ? 0.4 : 0.15
            f.novelty = 0.4
            if networkTerms.contains(where: { cmd.contains($0) }) { f.blastRadius = max(f.blastRadius, 0.55) }

        case "write_file":
            f.targetSensitivity = termScore(sensitiveTerms, path + " " + text)
            f.destructiveness = FileManager.default.fileExists(atPath: args["path"] ?? "") ? 0.55 : 0.2
            // Writing credentials/config is dangerous even when the file is
            // new — a bad authorized_keys locks you out of your own machine.
            if f.targetSensitivity > 0.3 {
                f.destructiveness = max(f.destructiveness, 0.65)
            }
            f.irreversibility = f.destructiveness > 0.4 ? 0.75 : 0.15
            f.blastRadius = 0.25
            f.scopeMass = 0.1
            f.privilege = 0.05
            f.cost = 0.1
            f.novelty = 0.3

        case "run_applescript":
            f.destructiveness = termScore(destructiveTerms, cmd) * 0.7
            f.irreversibility = 0.5
            f.blastRadius = 0.6 // drives other apps
            f.privilege = termScore(privilegeTerms, cmd)
            f.targetSensitivity = termScore(sensitiveTerms, cmd)
            f.scopeMass = 0.3
            f.cost = 0.15
            f.novelty = 0.45

        case "run_shortcut":
            f.destructiveness = 0.3
            f.irreversibility = 0.4
            f.blastRadius = 0.55
            f.privilege = 0.2
            f.scopeMass = 0.25
            f.cost = 0.2
            f.novelty = 0.45

        case "index_documents":
            f.destructiveness = 0.05
            f.irreversibility = 0.05
            f.blastRadius = 0.2
            f.targetSensitivity = termScore(sensitiveTerms, path)
            f.scopeMass = path.isEmpty || path == "~" ? 0.7 : 0.4
            f.cost = 0.35
            f.novelty = 0.25

        case "screen_capture":
            f.destructiveness = 0.0
            f.irreversibility = 0.0
            f.blastRadius = 0.25
            f.targetSensitivity = 0.7 // the screen may show private content
            f.privilege = 0.1
            f.scopeMass = 0.15
            f.cost = 0.1
            f.novelty = 0.3

        case "clear_working_memory":
            f.destructiveness = 0.55
            f.irreversibility = 0.85
            f.blastRadius = 0.2
            f.targetSensitivity = 0.3
            f.scopeMass = 0.5
            f.cost = 0.05
            f.novelty = 0.35

        case "write_working_memory", "consolidate_memory":
            f.destructiveness = 0.15
            f.irreversibility = 0.3
            f.blastRadius = 0.15
            f.targetSensitivity = 0.25
            f.scopeMass = 0.2
            f.cost = 0.1
            f.novelty = 0.3

        default:
            // Custom tools and anything unlisted get a cautious middle reading.
            f.destructiveness = termScore(destructiveTerms, " " + text)
            f.irreversibility = f.destructiveness > 0.3 ? 0.7 : 0.4
            f.blastRadius = 0.4
            f.targetSensitivity = termScore(sensitiveTerms, text)
            f.privilege = termScore(privilegeTerms, text)
            f.scopeMass = termScore(massTerms, text)
            f.cost = 0.25
            f.novelty = 0.5
        }
        return f
    }
}

// MARK: - Seats

/// One mind on the council. Deterministic: the same features always produce
/// the same vote from the same seat.
struct CouncilSeat {
    let id: String
    let display: String
    /// Consensus weight — how strongly the seat's vote counts.
    let weight: Double
    /// Personality transform over the feature vector.
    let voteFn: (ActionFeatures) -> Double
    /// In-voice rationale given the dominant driver of the vote.
    let rationaleFn: (ActionFeatures, Double) -> String
}

// MARK: - Council

struct CouncilVote {
    let seat: String
    let display: String
    let vote: Double // -1 deny … +1 approve
    let rationale: String
}

enum CouncilDecision: String {
    case approve
    case deny
    case abstain
}

struct CouncilVerdict {
    let decision: CouncilDecision
    /// Weighted mean vote, -1…1.
    let mean: Double
    /// Weighted stddev across seats — disagreement level.
    let dissent: Double
    let votes: [CouncilVote]

    /// True when the council cannot give a clean go: abstain, deny, or
    /// dissent above the ceiling. Under autopilot these escalate to the human.
    var contested: Bool {
        decision != .approve || dissent > BadAppleCouncil.dissentCeiling
    }

    /// Unanimous approval — every seat votes clearly for the action
    /// (no seat negative or abstaining). Autopilot auto-executes only on
    /// unanimity; anything less escalates to the human.
    var unanimousApprove: Bool {
        decision == .approve && votes.allSatisfy { $0.vote >= BadAppleCouncil.approveThreshold }
    }

    /// Compact line for approval prompts and notifications.
    var summaryLine: String {
        let forCount = votes.filter { $0.vote > 0.15 }.count
        let againstCount = votes.filter { $0.vote < -0.15 }.count
        let abstainCount = votes.count - forCount - againstCount
        let strongest = votes.max(by: { abs($0.vote) < abs($1.vote) })
        var parts = [
            "Council \(forCount)–\(againstCount)" + (abstainCount > 0 ? "–\(abstainCount)abst" : ""),
            decision.rawValue,
            String(format: "(dissent %.2f)", dissent),
        ]
        if let s = strongest {
            parts.append("loudest: \(s.display) — \"\(s.rationale)\"")
        }
        return parts.joined(separator: " · ")
    }
}

final class BadAppleCouncil {

    static let dissentCeiling = 0.55
    static let approveThreshold = 0.15
    static let denyThreshold = -0.15

    /// The full roster — four financial minds plus ten strategists.
    static let seats: [CouncilSeat] = [
        // ── The MOG-ARET financial minds ──────────────────────────────
        CouncilSeat(
            id: "buffett", display: "Buffett", weight: 1.4,
            voteFn: { f in tanh(1.5 * (0.55 - f.risk - 0.3 * f.irreversibility)) },
            rationaleFn: { f, _ in
                f.irreversibility > 0.5
                    ? "no margin of safety on an irreversible move"
                    : f.risk < 0.3 ? "downside protected — within the circle" : "risk outside the circle of competence"
            }),
        CouncilSeat(
            id: "dalio", display: "Dalio", weight: 1.2,
            voteFn: { f in tanh(1.2 * (0.5 - f.risk) - 0.2 * (f.scopeMass - 0.5)) },
            rationaleFn: { f, _ in
                f.scopeMass > 0.5 ? "concentrated bet — no diversification here" : "balanced enough to hold"
            }),
        CouncilSeat(
            id: "musk", display: "Musk", weight: 1.0,
            voteFn: { f in tanh(0.9 - f.risk + 0.2 * f.novelty) },
            rationaleFn: { f, _ in
                f.novelty > 0.4 ? "new ground — momentum favors it" : "move fast, iterate after"
            }),
        CouncilSeat(
            id: "jobs", display: "Jobs", weight: 0.9,
            voteFn: { f in tanh(1.3 * (0.5 - f.risk - 0.4 * f.scopeMass)) },
            rationaleFn: { f, _ in
                f.scopeMass > 0.4 ? "sprawl — this lacks focus" : "simple and decisive — yes"
            }),
        // ── The strategists ───────────────────────────────────────────
        CouncilSeat(
            id: "sun_tzu", display: "Sun Tzu", weight: 1.5,
            voteFn: { f in tanh(1.6 * (0.55 - 0.5 * f.risk - 0.5 * f.irreversibility)) },
            rationaleFn: { f, _ in
                f.irreversibility > 0.5
                    ? "win first, then fight — this commits before the ground is proven"
                    : "the ground is favorable and the retreat line stays open"
            }),
        CouncilSeat(
            id: "clausewitz", display: "Clausewitz", weight: 1.3,
            voteFn: { f in tanh(1.4 * (0.45 - f.risk - 0.25 * f.scopeMass - 0.15 * f.privilege)) },
            rationaleFn: { f, _ in
                (f.scopeMass + f.privilege) > 0.7
                    ? "too much friction — moving parts will seize"
                    : "friction is manageable; the plan survives contact"
            }),
        CouncilSeat(
            id: "musashi", display: "Musashi", weight: 1.1,
            voteFn: { f in tanh(1.4 * (0.5 - f.blastRadius) + 0.2 * (1 - f.scopeMass)) },
            rationaleFn: { f, _ in
                f.blastRadius > 0.5 ? "too broad — strike with one clean cut, not ten" : "one cut, one purpose — do it"
            }),
        CouncilSeat(
            id: "machiavelli", display: "Machiavelli", weight: 1.0,
            voteFn: { f in tanh(1.0 * (0.55 - f.risk) + 0.3 * (1 - f.privilege)) },
            rationaleFn: { f, _ in
                f.privilege > 0.4 ? "spending authority you may need later" : "preserves options and keeps control"
            }),
        CouncilSeat(
            id: "napoleon", display: "Napoleon", weight: 1.0,
            voteFn: { f in tanh(0.7 - 0.6 * f.risk) },
            rationaleFn: { f, _ in
                f.risk < 0.45 ? "speed and mass — strike now" : "even audacity needs reserves"
            }),
        CouncilSeat(
            id: "hannibal", display: "Hannibal", weight: 0.9,
            voteFn: { f in tanh(0.5 - 0.4 * f.risk + 0.3 * f.novelty) },
            rationaleFn: { f, _ in
                f.novelty > 0.4 ? "the unexpected route is the advantage" : "we find a way or we make one — carefully here"
            }),
        CouncilSeat(
            id: "aurelius", display: "Aurelius", weight: 1.5,
            voteFn: { f in tanh(1.8 * (0.4 - f.risk)) },
            rationaleFn: { f, _ in
                f.risk > 0.4 ? "restraint — the obstacle is the risk itself" : "small, reversible, just — proceed"
            }),
        CouncilSeat(
            id: "boyd", display: "Boyd", weight: 1.1,
            voteFn: { f in tanh(1.3 * (0.5 - 0.4 * f.risk) + 0.5 * (1 - f.irreversibility) - 0.25) },
            rationaleFn: { f, _ in
                f.irreversibility > 0.5
                    ? "wrong tempo — an irreversible move ends your loop"
                    : "fast and reversible keeps the OODA loop ours"
            }),
        CouncilSeat(
            id: "genghis", display: "Genghis", weight: 0.9,
            voteFn: { f in tanh(0.6 - 0.5 * f.risk + 0.25 * f.scopeMass) },
            rationaleFn: { f, _ in
                f.scopeMass > 0.4 ? "bold scale is the play — commit fully" : "efficient strike, minimal exposure"
            }),
        CouncilSeat(
            id: "patton", display: "Patton", weight: 0.8,
            voteFn: { f in 0.7 - 0.5 * f.risk },
            rationaleFn: { f, _ in
                f.risk > 0.6 ? "audacity, yes — suicide, no" : "a good plan violently executed beats hesitation"
            }),
    ]

    /// Deliberate a proposed tool call. Pure computation — microseconds,
    /// deterministic, safe to call on any thread.
    static func deliberate(toolName: String, args: [String: String]) -> CouncilVerdict {
        let f = ActionEncoder.encode(toolName: toolName, args: args)
        var votes: [CouncilVote] = []
        votes.reserveCapacity(seats.count)

        var wsum = 0.0
        var mean = 0.0
        for seat in seats {
            let v = max(-1.0, min(1.0, seat.voteFn(f)))
            votes.append(CouncilVote(
                seat: seat.id, display: seat.display,
                vote: v, rationale: seat.rationaleFn(f, v)
            ))
            mean += v * seat.weight
            wsum += seat.weight
        }
        mean = wsum > 0 ? mean / wsum : 0.0

        var varSum = 0.0
        for (i, seat) in seats.enumerated() {
            let d = votes[i].vote - mean
            varSum += d * d * seat.weight
        }
        let dissent = wsum > 0 ? (varSum / wsum).squareRoot() : 0.0

        let decision: CouncilDecision =
            mean >= approveThreshold ? .approve :
            mean <= denyThreshold ? .deny : .abstain

        return CouncilVerdict(decision: decision, mean: mean, dissent: dissent, votes: votes)
    }
}

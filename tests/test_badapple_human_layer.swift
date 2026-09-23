import Foundation

@main
enum BadAppleHumanLayerTests {
    static func main() {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appendingPathComponent("badapple-human-test-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: dir) }
        let layer = BadAppleHumanLayer(directory: dir)
        let now = Date()

        guard layer.homeText(now: now) == "Human Home is clear. Nothing is waiting on us." else {
            fail("default empty home")
        }
        guard layer.promptContext(now: now).isEmpty else {
            fail("empty prompt context")
        }

        do {
            _ = try layer.rememberPreference(key: "Morning Drink", value: "coffee")
            _ = try layer.rememberPreference(key: "morning drink", value: "tea")
        } catch { fail("rememberPreference threw: \(error)") }
        let prefs = layer.snapshot().preferences
        guard prefs.count == 1, prefs[0].key == "Morning Drink", prefs[0].value == "tea" else {
            fail("case-insensitive preference upsert")
        }
        do {
            _ = try layer.rememberPreference(key: "  ", value: "x")
            fail("empty preference key should throw")
        } catch {}

        let userCommitment: BadAppleHumanCommitment
        do {
            userCommitment = try layer.addCommitment(
                title: "Call mom", owner: .user, dueAt: nil, threadID: nil
            )
            _ = try layer.addCommitment(
                title: "Watch the build", owner: .badApple, dueAt: nil, threadID: nil
            )
            _ = try layer.addCommitment(
                title: "Pay the bill", owner: .user,
                dueAt: now.addingTimeInterval(-3600), threadID: nil
            )
        } catch { fail("addCommitment threw: \(error)") }
        var home = layer.homeText(now: now)
        guard home.contains("NOW"), home.contains("Pay the bill") else {
            fail("NOW section placement")
        }
        guard home.contains("WAITING ON YOU"), home.contains("Call mom") else {
            fail("WAITING ON YOU section")
        }
        guard home.contains("I'M HANDLING"), home.contains("Watch the build") else {
            fail("I'M HANDLING section")
        }
        guard home.contains("REMEMBERING"), home.contains("Morning Drink: tea") else {
            fail("REMEMBERING section")
        }

        do {
            let prefix = String(userCommitment.id.prefix(8))
            let updated = try layer.updateCommitment(id: prefix, status: .completed)
            guard updated?.id == userCommitment.id, updated?.status == .completed else {
                fail("unique-prefix completion")
            }
            guard try layer.updateCommitment(id: "nonexistent000", status: .completed) == nil else {
                fail("updateCommitment unknown id should return nil")
            }
        } catch { fail("updateCommitment threw: \(error)") }
        home = layer.homeText(now: now)
        guard !home.contains("Call mom") else {
            fail("completed commitment still listed")
        }

        do {
            let thread = try layer.upsertThread(
                id: nil, title: "Move to Portland",
                summary: "Packing", nextAction: "Book movers", status: .active
            )
            let updated = try layer.upsertThread(
                id: String(thread.id.prefix(8)), title: "",
                summary: "Packing done", nextAction: "Sign lease", status: .waiting
            )
            guard updated.id == thread.id,
                  updated.title == "Move to Portland",
                  updated.summary == "Packing done",
                  updated.nextAction == "Sign lease",
                  updated.status == .waiting else {
                fail("thread upsert")
            }
        } catch { fail("upsertThread threw: \(error)") }
        guard layer.homeText(now: now).contains("LIFE THREADS"),
              layer.homeText(now: now).contains("Sign lease") else {
            fail("LIFE THREADS section")
        }

        let second = BadAppleHumanLayer(directory: dir)
        let snap = second.snapshot()
        guard snap.preferences.count == 1,
              snap.commitments.count == 3,
              snap.threads.count == 1 else {
            fail("persistence across instances")
        }

        let validFixtures = [
            "today", "tomorrow", "TODAY 9am", "tomorrow 14:30", "today 9:30pm",
            "in 30 minutes", "in 1 minute", "In 2 Hours", "in 3 days",
            "2030-01-02 03:04", "2030-01-02", "2030-01-02T03:04:05Z",
        ]
        for fixture in validFixtures {
            guard BadAppleHumanLayer.parseDueDate(fixture, now: now) != nil else {
                fail("parseDueDate valid fixture '\(fixture)'")
            }
        }
        for fixture in ["", "   ", "next friday-ish", "tomorrow banana", "in a while"] {
            guard BadAppleHumanLayer.parseDueDate(fixture, now: now) == nil else {
                fail("parseDueDate invalid fixture '\(fixture)'")
            }
        }
        if let parsed = BadAppleHumanLayer.parseDueDate("tomorrow 14:30", now: now) {
            let comps = Calendar.current.dateComponents([.hour, .minute], from: parsed)
            guard comps.hour == 14, comps.minute == 30,
                  Calendar.current.isDateInTomorrow(parsed) else {
                fail("tomorrow 14:30 lands on tomorrow at 14:30")
            }
        }
        if let parsed = BadAppleHumanLayer.parseDueDate("in 2 hours", now: now) {
            let delta = parsed.timeIntervalSince(now)
            guard abs(delta - 7200) < 60 else {
                fail("in 2 hours offset")
            }
        }

        do {
            let first = try layer.claimDueCommitments(now: now)
            guard first.count == 1, first[0].title == "Pay the bill" else {
                fail("claimDueCommitments initial claim")
            }
            let again = try layer.claimDueCommitments(now: now.addingTimeInterval(3600))
            guard again.isEmpty else {
                fail("claimDueCommitments re-notified within 24h")
            }
            let later = try layer.claimDueCommitments(now: now.addingTimeInterval(25 * 3600))
            guard later.count == 1 else {
                fail("claimDueCommitments re-notifies after 24h")
            }
        } catch { fail("claimDueCommitments threw: \(error)") }

        func event(importance: Int, urgency: Int, requiresAction: Bool, voice: Bool) -> BadAppleHumanEvent {
            BadAppleHumanEvent(
                id: UUID().uuidString, kind: "test", title: "t", body: "b",
                importance: importance, urgency: urgency,
                requiresAction: requiresAction, voiceRequested: voice,
                createdAt: now
            )
        }
        do {
            try layer.setAttentionMode(.available)
            guard layer.route(event: event(importance: 3, urgency: 4, requiresAction: true, voice: true)).channel == .speak,
                  layer.route(event: event(importance: 3, urgency: 4, requiresAction: true, voice: false)).channel == .notify,
                  layer.route(event: event(importance: 1, urgency: 1, requiresAction: false, voice: false)).channel == .silent else {
                fail("route: available mode")
            }
            try layer.setAttentionMode(.focus)
            guard layer.route(event: event(importance: 3, urgency: 2, requiresAction: true, voice: false)).channel == .notify,
                  layer.route(event: event(importance: 3, urgency: 5, requiresAction: false, voice: false)).channel == .speak,
                  layer.route(event: event(importance: 3, urgency: 2, requiresAction: false, voice: false)).channel == .queue else {
                fail("route: focus mode")
            }
            try layer.setAttentionMode(.quiet)
            guard layer.route(event: event(importance: 3, urgency: 4, requiresAction: true, voice: false)).channel == .notify,
                  layer.route(event: event(importance: 3, urgency: 4, requiresAction: false, voice: false)).channel == .queue,
                  layer.route(event: event(importance: 3, urgency: 5, requiresAction: false, voice: false)).channel == .speak else {
                fail("route: quiet mode")
            }
            try layer.setAttentionMode(.sleep)
            guard layer.route(event: event(importance: 3, urgency: 5, requiresAction: true, voice: false)).channel == .speak,
                  layer.route(event: event(importance: 4, urgency: 4, requiresAction: true, voice: true)).channel == .queue else {
                fail("route: sleep mode")
            }
            try layer.setAttentionMode(.available)
        } catch { fail("setAttentionMode threw: \(error)") }

        do {
            let held = BadAppleHumanEvent(
                id: "held-event-1", kind: "task_done", title: "Task finished",
                body: "lint pass", importance: 2, urgency: 2,
                requiresAction: false, voiceRequested: false, createdAt: now
            )
            let decision = layer.route(event: held)
            try layer.enqueue(event: held, decision: decision)
            try layer.enqueue(event: held, decision: decision)
            let homeNow = layer.homeText(now: now)
            guard homeNow.contains("HELD FOR LATER"), homeNow.contains("Task finished") else {
                fail("HELD FOR LATER section")
            }
            guard layer.snapshot().inbox.filter({ $0.event.id == "held-event-1" }).count == 1 else {
                fail("enqueue dedupes by event id")
            }
        } catch { fail("enqueue threw: \(error)") }

        let context = layer.promptContext(now: now)
        guard context.contains("attention_mode:"), context.contains("Morning Drink"),
              context.contains("commitments:"), context.contains("life_threads:") else {
            fail("promptContext content")
        }

        do {
            let older = BadAppleHumanEvent(
                id: "held-older", kind: "task_done", title: "Older task",
                body: "first", importance: 2, urgency: 2,
                requiresAction: false, voiceRequested: false,
                createdAt: now.addingTimeInterval(-120)
            )
            let newer = BadAppleHumanEvent(
                id: "held-newer", kind: "task_done", title: "Newer task",
                body: "second", importance: 2, urgency: 2,
                requiresAction: false, voiceRequested: false,
                createdAt: now.addingTimeInterval(-60)
            )
            let decision = layer.route(event: older)
            try layer.enqueue(event: older, decision: decision)
            try layer.enqueue(event: newer, decision: decision)
            let claimed = try layer.claimHeldEvents(limit: 20)
            guard claimed.map(\.id) == ["held-older", "held-newer", "held-event-1"] else {
                fail("claimHeldEvents oldest-first order: \(claimed.map(\.id))")
            }
            let homeAfterClaim = layer.homeText(now: now)
            guard !homeAfterClaim.contains("HELD FOR LATER"),
                  !homeAfterClaim.contains("Task finished") else {
                fail("claimed events still shown in HELD FOR LATER")
            }
            guard layer.snapshot().inbox.allSatisfy({ $0.status == .completed }) else {
                fail("claimed events not marked completed")
            }
            guard try layer.claimHeldEvents().isEmpty else {
                fail("second claimHeldEvents returned events")
            }
        } catch { fail("claimHeldEvents threw: \(error)") }

        let noopDir = fm.temporaryDirectory
            .appendingPathComponent("badapple-human-noop-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: noopDir) }
        let noopLayer = BadAppleHumanLayer(directory: noopDir)
        let noopStatePath = noopDir.appendingPathComponent("state.json").path
        do {
            _ = try noopLayer.rememberPreference(key: "mtime-check", value: "1")
            let before = try fm.attributesOfItem(atPath: noopStatePath)[.modificationDate] as? Date
            guard before != nil else { fail("no state file written") }
            usleep(1_100_000)
            _ = try noopLayer.claimDueCommitments(now: now)
            _ = try noopLayer.updateCommitment(id: "missing-id", status: .completed)
            try noopLayer.setAttentionMode(.available)
            let after = try fm.attributesOfItem(atPath: noopStatePath)[.modificationDate] as? Date
            guard before == after else {
                fail("no-op mutations rewrote state.json")
            }
        } catch { fail("no-op mutation check: \(error)") }

        let concDir = fm.temporaryDirectory
            .appendingPathComponent("badapple-human-conc-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: concDir) }
        let layerA = BadAppleHumanLayer(directory: concDir)
        let layerB = BadAppleHumanLayer(directory: concDir)
        let group = DispatchGroup()
        for (writer, tag) in [(layerA, "a"), (layerB, "b")] {
            group.enter()
            DispatchQueue.global().async {
                for i in 0..<50 {
                    _ = try? writer.rememberPreference(key: "conc-\(tag)-\(i)", value: "v\(i)")
                }
                group.leave()
            }
        }
        _ = group.wait(timeout: .now() + 60)
        let allPrefs = layerA.snapshot().preferences
        let allKeys = Set(allPrefs.map(\.key))
        guard allPrefs.count == 100, allKeys.count == 100 else {
            fail("cross-instance concurrent writes lost updates: \(allPrefs.count)")
        }

        let badDir = fm.temporaryDirectory
            .appendingPathComponent("badapple-human-bad-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: badDir) }
        let badLayer = BadAppleHumanLayer(directory: badDir)
        let badStateURL = badDir.appendingPathComponent("state.json")
        do {
            try "this is not json {".write(to: badStateURL, atomically: true, encoding: .utf8)
            guard badLayer.snapshot().preferences.isEmpty,
                  badLayer.snapshot().attentionMode == .available else {
                fail("malformed state falls back to empty")
            }
            let raw = try String(contentsOf: badStateURL, encoding: .utf8)
            guard raw == "this is not json {" else {
                fail("malformed state overwritten by a read")
            }
        } catch { fail("malformed state fixture: \(error)") }

        let statePath = dir.appendingPathComponent("state.json").path
        if let attrs = try? fm.attributesOfItem(atPath: statePath),
           let perms = attrs[.posixPermissions] as? NSNumber {
            guard perms.intValue & 0o777 == 0o600 else {
                fail("state file permissions \(String(perms.intValue & 0o777, radix: 8)) != 600")
            }
        } else {
            fail("state file attributes unreadable")
        }

        print("Bad Apple human layer tests passed")
    }

    private static func fail(_ name: String) -> Never {
        fputs("FAIL: \(name)\n", stderr)
        exit(1)
    }
}

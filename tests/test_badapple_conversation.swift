import Foundation

@main
enum BadAppleConversationTests {
    static func main() {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appendingPathComponent("badapple-conv-test-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: dir) }
        let store = BadAppleConversation(directory: dir)

        store.saveConversation(sessionId: "default", messages: [
            BadAppleMessage(role: "user", content: "hi"),
            BadAppleMessage(role: "assistant", content: "hello"),
        ])
        store.saveConversation(sessionId: "thread-abc", messages: [
            BadAppleMessage(role: "user", content: "work"),
        ])

        guard store.listSessionIDs() == ["default", "thread-abc"] else {
            fail("listSessionIDs sorted basenames: \(store.listSessionIDs())")
        }
        guard store.loadConversation(sessionId: "default").count == 2,
              store.loadConversation(sessionId: "thread-abc").count == 1 else {
            fail("separate sessions load independently")
        }

        store.clearConversation(sessionId: "default")
        guard store.loadConversation(sessionId: "default").isEmpty,
              store.loadConversation(sessionId: "thread-abc").count == 1 else {
            fail("clearing one session must not clear another")
        }
        guard store.listSessionIDs() == ["thread-abc"] else {
            fail("listSessionIDs after clear")
        }

        if let attrs = try? fm.attributesOfItem(atPath: dir.path),
           let perms = attrs[.posixPermissions] as? NSNumber {
            guard perms.intValue & 0o777 == 0o700 else {
                fail("conversations directory permissions \(String(perms.intValue & 0o777, radix: 8)) != 700")
            }
        } else {
            fail("conversations directory attributes unreadable")
        }
        let filePath = dir.appendingPathComponent("thread-abc.json").path
        if let attrs = try? fm.attributesOfItem(atPath: filePath),
           let perms = attrs[.posixPermissions] as? NSNumber {
            guard perms.intValue & 0o777 == 0o600 else {
                fail("conversation file permissions \(String(perms.intValue & 0o777, radix: 8)) != 600")
            }
        } else {
            fail("conversation file attributes unreadable")
        }

        let hostile = store.conversationPath(sessionId: "../evil")
        guard hostile.deletingLastPathComponent().path == dir.path,
              !hostile.lastPathComponent.contains("..") else {
            fail("session id sanitisation")
        }

        let longMessages = (0..<50).map {
            BadAppleMessage(role: "user", content: "m\($0)")
        }
        store.saveConversation(sessionId: "big", messages: longMessages)
        guard store.loadConversation(sessionId: "big").count == BadAppleConversation.maxPersistedMessages else {
            fail("persisted messages trimmed to cap")
        }

        let legacyDir = fm.temporaryDirectory
            .appendingPathComponent("badapple-conv-legacy-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: legacyDir) }
        do {
            try fm.createDirectory(
                at: legacyDir,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o755]
            )
            let legacyFile = legacyDir.appendingPathComponent("legacy.json")
            try Data("[{\"role\":\"user\",\"content\":\"old\"}]".utf8)
                .write(to: legacyFile)
            try fm.setAttributes([.posixPermissions: 0o644], ofItemAtPath: legacyFile.path)
        } catch {
            fail("legacy fixture setup: \(error)")
        }
        _ = BadAppleConversation(directory: legacyDir)
        if let attrs = try? fm.attributesOfItem(atPath: legacyDir.path),
           let perms = attrs[.posixPermissions] as? NSNumber {
            guard perms.intValue & 0o777 == 0o700 else {
                fail("init migrates directory permissions \(String(perms.intValue & 0o777, radix: 8)) != 700")
            }
        } else {
            fail("legacy directory attributes unreadable")
        }
        let legacyFilePath = legacyDir.appendingPathComponent("legacy.json").path
        if let attrs = try? fm.attributesOfItem(atPath: legacyFilePath),
           let perms = attrs[.posixPermissions] as? NSNumber {
            guard perms.intValue & 0o777 == 0o600 else {
                fail("init migrates legacy JSON permissions \(String(perms.intValue & 0o777, radix: 8)) != 600")
            }
        } else {
            fail("legacy JSON attributes unreadable")
        }

        let concurrentDir = fm.temporaryDirectory
            .appendingPathComponent("badapple-conv-conc-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: concurrentDir) }
        let storeA = BadAppleConversation(directory: concurrentDir)
        let storeB = BadAppleConversation(directory: concurrentDir)
        DispatchQueue.concurrentPerform(iterations: 20) { i in
            let s = i < 10 ? storeA : storeB
            s.appendTurn(
                sessionId: "shared",
                prompt: "prompt-\(i)",
                response: "response-\(i)"
            )
        }
        let merged = storeA.loadConversation(sessionId: "shared")
        guard merged.count == 40 else {
            fail("concurrent appendTurn lost writes: \(merged.count) messages")
        }
        guard merged.filter({ $0.role == "user" }).count == 20,
              merged.filter({ $0.role == "assistant" }).count == 20 else {
            fail("concurrent appendTurn produced wrong role mix")
        }
        for i in 0..<20 {
            guard merged.contains(where: { $0.content == "prompt-\(i)" }) else {
                fail("missing prompt-\(i) after concurrent append")
            }
        }

        print("Bad Apple conversation tests passed")
    }

    private static func fail(_ name: String) -> Never {
        fputs("FAIL: \(name)\n", stderr)
        exit(1)
    }
}

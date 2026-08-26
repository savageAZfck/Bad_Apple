import Foundation
import AppKit

/// Native glass control center for the Bad Apple AI OS.
/// Lives in the menu bar and uses the embedded `badapple` binary for control.
final class BadAppleControlCenter: NSObject, NSWindowDelegate {
    private var window: NSWindow?
    private let visualEffectView = NSVisualEffectView()
    private let contentStack = NSStackView()

    private let statusLabel = NSTextField(labelWithString: "Status: —")
    private let modeLabel = NSTextField(labelWithString: "Mode: —")
    private let workspaceLabel = NSTextField(labelWithString: "Workspace: —")
    private let memoryLabel = NSTextField(labelWithString: "Memory: —")
    private let modelLabel = NSTextField(labelWithString: "Model: —")
    private let p2pLabel = NSTextField(labelWithString: "P2P: —")
    private let hibernationLabel = NSTextField(labelWithString: "Hibernating: —")
    private let statusDot = NSView()

    private let fastTierButton = NSButton()
    private let autopilotButton = NSButton()
    private let p2pButton = NSButton()
    private let voiceButton = NSButton()
    private let purgeButton = NSButton()
    private let unloadButton = NSButton()
    private let killButton = NSButton()

    private var refreshTimer: Timer?
    private var running = false

    private var fastTier = false
    private var autopilot = false
    private var p2p = false
    private var voice = false
    private var killed = false

    var onVoiceToggle: ((Bool) -> Void)?

    override init() {
        super.init()
        buildWindow()
    }

    deinit {
        refreshTimer?.invalidate()
    }

    private func buildWindow() {
        visualEffectView.material = .hudWindow
        visualEffectView.blendingMode = .behindWindow
        visualEffectView.state = .active
        visualEffectView.translatesAutoresizingMaskIntoConstraints = false

        contentStack.orientation = .vertical
        contentStack.alignment = .leading
        contentStack.spacing = 12
        contentStack.translatesAutoresizingMaskIntoConstraints = false
        visualEffectView.addSubview(contentStack)

        let title = NSTextField(labelWithString: "Bad Apple Control Center")
        title.font = NSFont.systemFont(ofSize: 18, weight: .semibold)
        title.textColor = NSColor.labelColor
        contentStack.addArrangedSubview(title)

        let statusRow = NSStackView()
        statusRow.orientation = .horizontal
        statusRow.spacing = 8
        statusRow.alignment = .centerY
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 6
        statusDot.layer?.masksToBounds = true
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            statusDot.widthAnchor.constraint(equalToConstant: 12),
            statusDot.heightAnchor.constraint(equalToConstant: 12)
        ])
        statusRow.addArrangedSubview(statusDot)
        statusLabel.font = NSFont.systemFont(ofSize: 13, weight: .regular)
        statusLabel.textColor = NSColor.secondaryLabelColor
        statusRow.addArrangedSubview(statusLabel)
        contentStack.addArrangedSubview(statusRow)

        for label in [modeLabel, workspaceLabel, memoryLabel, modelLabel, p2pLabel, hibernationLabel] {
            label.font = NSFont.systemFont(ofSize: 13, weight: .regular)
            label.textColor = NSColor.secondaryLabelColor
            contentStack.addArrangedSubview(label)
        }

        contentStack.setCustomSpacing(20, after: hibernationLabel)

        let toggles = [fastTierButton, autopilotButton, p2pButton, voiceButton]
        for button in toggles {
            button.setButtonType(.toggle)
            button.font = NSFont.systemFont(ofSize: 13)
            contentStack.addArrangedSubview(button)
        }

        fastTierButton.title = "Fast Tier Only"
        fastTierButton.toolTip = "Route simple queries to the 0.5B fast model."
        fastTierButton.target = self
        fastTierButton.action = #selector(toggleFastTier)

        autopilotButton.title = "Autopilot"
        autopilotButton.toolTip = "Allow destructive tools to run without approval prompts."
        autopilotButton.target = self
        autopilotButton.action = #selector(toggleAutopilot)

        p2pButton.title = "P2P Sync"
        p2pButton.toolTip = "Enable or disable encrypted link-local peer discovery and sync."
        p2pButton.target = self
        p2pButton.action = #selector(toggleP2P)

        voiceButton.title = "Voice"
        voiceButton.toolTip = "Toggle voice listening and TTS output."
        voiceButton.target = self
        voiceButton.action = #selector(toggleVoice)

        let buttonStack = NSStackView()
        buttonStack.orientation = .horizontal
        buttonStack.spacing = 8
        contentStack.addArrangedSubview(buttonStack)

        for button in [purgeButton, unloadButton, killButton] {
            button.bezelStyle = .rounded
            button.font = NSFont.systemFont(ofSize: 12)
            buttonStack.addArrangedSubview(button)
        }

        purgeButton.title = "Purge VRAM"
        purgeButton.toolTip = "Release cached GPU memory and Metal allocations."
        purgeButton.target = self
        purgeButton.action = #selector(purgeVRAM)

        unloadButton.title = "Unload Models"
        unloadButton.toolTip = "Unload vision, image, and optional models to free RAM."
        unloadButton.target = self
        unloadButton.action = #selector(unloadModels)

        killButton.title = "Stop"
        killButton.toolTip = "Stop generation and block tools until resumed."
        killButton.target = self
        killButton.action = #selector(toggleKill)

        let dashboard = NSButton(title: "Open Dashboard", target: self, action: #selector(openDashboard))
        dashboard.toolTip = "Open the Bad Apple web dashboard in your browser."
        dashboard.bezelStyle = .rounded
        contentStack.addArrangedSubview(dashboard)

        let controlRect = NSRect(x: 0, y: 0, width: 340, height: 420)
        visualEffectView.frame = controlRect

        let w = NSPanel(contentRect: controlRect, styleMask: [.titled, .closable, .hudWindow, .utilityWindow], backing: .buffered, defer: false)
        w.title = "Bad Apple"
        w.isFloatingPanel = true
        w.hidesOnDeactivate = false
        w.contentView = visualEffectView
        w.delegate = self
        w.titlebarAppearsTransparent = true
        w.isMovableByWindowBackground = true
        w.level = NSWindow.Level.floating
        window = w

        NSLayoutConstraint.activate([
            contentStack.topAnchor.constraint(equalTo: visualEffectView.topAnchor, constant: 24),
            contentStack.leadingAnchor.constraint(equalTo: visualEffectView.leadingAnchor, constant: 20),
            contentStack.trailingAnchor.constraint(equalTo: visualEffectView.trailingAnchor, constant: -20),
            contentStack.bottomAnchor.constraint(lessThanOrEqualTo: visualEffectView.bottomAnchor, constant: -20),
        ])
    }

    func show() {
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        startRefreshing()
    }

    func windowWillClose(_ notification: Notification) {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }

    private func startRefreshing() {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
        refresh()
    }

    private func refresh() {
        Task {
            do {
                let output = try await runCLI(args: ["runtime", "status"])
                await update(from: output)
            } catch {
                await MainActor.run {
                    statusLabel.stringValue = "Status: unreachable"
                    statusDot.layer?.backgroundColor = NSColor.systemRed.cgColor
                }
            }
        }
    }

    @MainActor
    private func update(from output: String) {
        guard let data = output.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            statusLabel.stringValue = "Status: invalid"
            statusDot.layer?.backgroundColor = NSColor.systemRed.cgColor
            return
        }

        let runtime = json["runtime"] as? [String: Any]
        let mode = runtime?["mode"] as? String ?? "—"
        let killed = runtime?["killed"] as? Bool ?? false
        let safeReason = runtime?["safe_mode_reason"] as? String
        let healthColor: NSColor
        if killed || safeReason != nil || mode == "SAFE_MODE" {
            healthColor = .systemYellow
        } else if mode == "READY" {
            healthColor = .systemGreen
        } else if mode == "STARTING" {
            healthColor = .systemYellow
        } else {
            healthColor = .systemGray
        }
        statusDot.layer?.backgroundColor = healthColor.cgColor
        statusLabel.stringValue = "Status: \(mode)"
        modeLabel.stringValue = "Mode: \(mode)"
        self.killed = killed
        killButton.title = killed ? "Resume" : "Stop"

        if let mem = json["resources"] as? [String: Any] {
            let pct = mem["memory_percent"] as? Double ?? 0
            let gb = mem["available_gb"] as? Double ?? 0
            memoryLabel.stringValue = String(format: "Memory: %.1f%% (%.2f GB free)", pct, gb)
        }

        workspaceLabel.stringValue = "Workspace: \(json["workspace"] as? String ?? "—")"

        if let models = json["active_models"] as? [String] {
            modelLabel.stringValue = "Model: \(models.joined(separator: ", "))"
        }

        p2p = json["p2p_enabled"] as? Bool ?? false
        p2pButton.state = p2p ? .on : .off
        p2pLabel.stringValue = "P2P: \(p2p ? "on" : "off")"

        fastTier = json["fast_tier"] as? Bool ?? false
        fastTierButton.state = fastTier ? .on : .off

        autopilot = json["autopilot"] as? Bool ?? false
        autopilotButton.state = autopilot ? .on : .off

        voice = UserDefaults.standard.object(forKey: "BadAppleVoiceEnabled") as? Bool ?? true
        voiceButton.state = voice ? .on : .off

        let hibernating = json["hibernating"] as? Bool ?? false
        hibernationLabel.stringValue = "Hibernating: \(hibernating ? "yes" : "no")"
    }

    @objc private func toggleFastTier() {
        send(prompt: fastTierButton.state == .on ? "fast tier on" : "fast tier off")
    }

    @objc private func toggleAutopilot() {
        send(prompt: autopilotButton.state == .on ? "autopilot on" : "autopilot off")
    }

    @objc private func toggleP2P() {
        send(prompt: p2pButton.state == .on ? "p2p on" : "p2p off")
    }

    @objc private func toggleVoice() {
        onVoiceToggle?(voiceButton.state == .on)
    }

    @objc private func purgeVRAM() {
        send(prompt: "flush vram")
    }

    @objc private func unloadModels() {
        send(prompt: "unload all models")
    }

    @objc private func toggleKill() {
        if killed {
            send(prompt: "resume bad apple")
        } else {
            send(prompt: "kill switch")
        }
    }

    @objc private func openDashboard() {
        if let url = URL(string: "http://127.0.0.1:8787") {
            NSWorkspace.shared.open(url)
        }
    }

    private func send(prompt: String) {
        Task {
            _ = try? await runCLI(args: [prompt])
            refresh()
        }
    }

    private func runCLI(args: [String]) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                guard let helper = Bundle.main.url(forAuxiliaryExecutable: "badapple")?.path else {
                    continuation.resume(throwing: NSError(domain: "BadApple", code: 1, userInfo: [NSLocalizedDescriptionKey: "badapple binary not found"]))
                    return
                }
                let process = Process()
                process.executableURL = URL(fileURLWithPath: helper)
                process.arguments = args
                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = FileHandle.nullDevice
                do {
                    try process.run()
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    let output = String(data: data, encoding: .utf8) ?? ""
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

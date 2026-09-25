import AppKit

/// Native agent task board — a live view of the persistent agent queue with
/// pause/resume/cancel controls. Talks to the running daemon over the same
/// `__BADAPPLE_AGENT__` JSON channel the web Control Center uses, so it works
/// whether the engine lives in this process or in the LaunchDaemon.
final class BadAppleTasksWindow: NSObject {

    private struct TaskRow {
        let id: String
        let status: String
        let goal: String
        let steps: Int
        let maxSteps: Int
    }

    private var window: NSWindow?
    private var scrollView: NSScrollView?
    private var document: NSView?
    private var goalField: NSTextField?
    private var runButton: NSButton?
    private var emptyLabel: NSTextField?
    private var refreshTimer: Timer?
    private var tasks: [TaskRow] = []
    private var submitting = false

    func show() {
        if window == nil { buildWindow() }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        refresh()
        startPolling()
    }

    private func startPolling() {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            guard let self, self.window?.isVisible == true else { return }
            self.refresh()
        }
    }

    // MARK: UI

    private func buildWindow() {
        let w = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 560, height: 460),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false)
        w.title = "Agent Tasks"
        w.minSize = NSSize(width: 440, height: 300)
        w.isReleasedWhenClosed = false
        window = w
        guard let content = w.contentView else { return }

        let header = NSView()
        header.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(header)

        let title = NSTextField(labelWithString: "Tasks")
        title.font = .boldSystemFont(ofSize: 15)
        title.translatesAutoresizingMaskIntoConstraints = false
        header.addSubview(title)

        let field = NSTextField()
        field.placeholderString = "New task goal — she plans and runs it"
        field.translatesAutoresizingMaskIntoConstraints = false
        field.target = self
        field.action = #selector(submitTask(_:))
        header.addSubview(field)
        goalField = field

        let run = NSButton(title: "Run", target: self, action: #selector(submitTask(_:)))
        run.bezelStyle = .rounded
        run.keyEquivalent = "\r"
        run.translatesAutoresizingMaskIntoConstraints = false
        header.addSubview(run)
        runButton = run

        let refresh = NSButton(title: "↻", target: self, action: #selector(refreshNow(_:)))
        refresh.bezelStyle = .rounded
        refresh.translatesAutoresizingMaskIntoConstraints = false
        header.addSubview(refresh)

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        scroll.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(scroll)
        scrollView = scroll
        let doc = NSView()
        scroll.documentView = doc
        document = doc

        let empty = NSTextField(labelWithString: "No agent tasks yet — give her a goal above.")
        empty.textColor = .secondaryLabelColor
        empty.alignment = .center
        empty.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(empty)
        emptyLabel = empty

        NSLayoutConstraint.activate([
            header.topAnchor.constraint(equalTo: content.topAnchor),
            header.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            header.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            header.heightAnchor.constraint(equalToConstant: 52),

            title.leadingAnchor.constraint(equalTo: header.leadingAnchor, constant: 14),
            title.centerYAnchor.constraint(equalTo: header.centerYAnchor),

            field.leadingAnchor.constraint(equalTo: title.trailingAnchor, constant: 12),
            field.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            field.heightAnchor.constraint(equalToConstant: 24),

            run.leadingAnchor.constraint(equalTo: field.trailingAnchor, constant: 8),
            run.centerYAnchor.constraint(equalTo: header.centerYAnchor),

            refresh.leadingAnchor.constraint(equalTo: run.trailingAnchor, constant: 6),
            refresh.trailingAnchor.constraint(equalTo: header.trailingAnchor, constant: -10),
            refresh.centerYAnchor.constraint(equalTo: header.centerYAnchor),

            scroll.topAnchor.constraint(equalTo: header.bottomAnchor),
            scroll.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: content.bottomAnchor),

            empty.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            empty.centerYAnchor.constraint(equalTo: content.centerYAnchor, constant: 20),
        ])
    }

    private func rebuildRows() {
        guard let scroll = scrollView, let doc = document else { return }
        doc.subviews.forEach { $0.removeFromSuperview() }
        emptyLabel?.isHidden = !tasks.isEmpty
        let width = scroll.contentView.bounds.width
        var y: CGFloat = 8
        var rows: [NSView] = []
        // Newest last in the daemon's list; show newest on top.
        for task in tasks.reversed() {
            let row = makeRow(task, width: width)
            row.frame.origin = NSPoint(x: 8, y: y)
            doc.addSubview(row)
            rows.append(row)
            y += row.bounds.height + 8
        }
        let docH = max(y, scroll.contentView.bounds.height)
        doc.frame = NSRect(x: 0, y: 0, width: width, height: docH)
        // Document is bottom-left origin; flip so the first row sits on top.
        for v in doc.subviews {
            v.frame.origin.y = docH - v.frame.origin.y - v.bounds.height
        }
    }

    private func makeRow(_ task: TaskRow, width: CGFloat) -> NSView {
        let row = NSView()
        row.wantsLayer = true
        row.layer?.cornerRadius = 8
        row.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor

        let dot = NSView()
        dot.wantsLayer = true
        dot.layer?.cornerRadius = 5
        dot.layer?.backgroundColor = Self.statusColor(task.status).cgColor
        dot.frame = NSRect(x: 12, y: 14, width: 10, height: 10)
        row.addSubview(dot)

        let goal = NSTextField(wrappingLabelWithString: task.goal)
        goal.font = .systemFont(ofSize: 13)
        goal.lineBreakMode = .byTruncatingTail
        goal.maximumNumberOfLines = 2
        let goalW = width - 16 - 40
        let goalH = goal.sizeThatFits(NSSize(width: goalW, height: .greatestFiniteMagnitude)).height
        goal.frame = NSRect(x: 32, y: 0, width: goalW, height: goalH)
        row.addSubview(goal)

        let meta = NSTextField(labelWithString: "\(task.status) · steps \(task.steps)/\(task.maxSteps)")
        meta.font = .systemFont(ofSize: 11)
        meta.textColor = .secondaryLabelColor
        meta.sizeToFit()
        row.addSubview(meta)

        let contentH = goalH + 4 + meta.bounds.height
        row.frame.size = NSSize(width: width - 16, height: contentH + 16)
        goal.frame.origin.y = 8 + meta.bounds.height + 4
        meta.frame.origin = NSPoint(x: 32, y: 8)

        var bx = width - 16 - 10
        let buttons: [(String, Selector, [String])] = [
            ("Cancel", #selector(cancelTask(_:)), ["queued", "running", "paused", "waiting"]),
            ("Resume", #selector(resumeTask(_:)), ["paused", "waiting"]),
            ("Pause", #selector(pauseTask(_:)), ["running", "queued"]),
        ]
        var buttonHeight: CGFloat = 0
        for (title, action, statuses) in buttons where statuses.contains(task.status) {
            let b = NSButton(title: title, target: self, action: action)
            b.bezelStyle = .rounded
            b.controlSize = .small
            b.identifier = NSUserInterfaceItemIdentifier(task.id)
            b.sizeToFit()
            b.frame.origin = NSPoint(x: bx - b.bounds.width, y: 6)
            bx -= b.bounds.width + 6
            row.addSubview(b)
            buttonHeight = max(buttonHeight, b.bounds.height)
        }
        _ = buttonHeight

        return row
    }

    private static func statusColor(_ status: String) -> NSColor {
        switch status {
        case "running", "working": return .systemGreen
        case "queued": return .systemBlue
        case "paused", "waiting": return .systemYellow
        case "failed", "cancelled": return .systemRed
        default: return .systemGray
        }
    }

    // MARK: Actions

    @objc private func refreshNow(_ sender: Any?) { refresh() }

    @objc private func submitTask(_ sender: Any?) {
        guard !submitting, let field = goalField else { return }
        let goal = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !goal.isEmpty else { return }
        submitting = true
        runButton?.isEnabled = false
        Task { [weak self] in
            let _ = await self?.callAgent("run_agent_task", params: ["goal": goal, "max_steps": 10])
            await MainActor.run {
                self?.submitting = false
                self?.runButton?.isEnabled = true
                self?.goalField?.stringValue = ""
                self?.refresh()
            }
        }
    }

    @objc private func pauseTask(_ sender: NSButton) { taskAction("pause_agent_task", sender) }
    @objc private func resumeTask(_ sender: NSButton) { taskAction("resume_agent_task", sender) }
    @objc private func cancelTask(_ sender: NSButton) { taskAction("cancel_agent_task", sender) }

    private func taskAction(_ method: String, _ sender: NSButton) {
        guard let id = sender.identifier?.rawValue, !id.isEmpty else { return }
        sender.isEnabled = false
        Task { [weak self] in
            let _ = await self?.callAgent(method, params: ["task_id": id])
            await MainActor.run { self?.refresh() }
        }
    }

    private func refresh() {
        Task { [weak self] in
            guard let result = await self?.callAgent("list_agent_tasks", params: [:]),
                  let list = result["tasks"] as? [[String: Any]] else { return }
            let rows = list.compactMap { t -> TaskRow? in
                guard let id = t["id"] as? String else { return nil }
                return TaskRow(
                    id: id,
                    status: t["status"] as? String ?? "unknown",
                    goal: t["goal"] as? String ?? "",
                    steps: t["steps"] as? Int ?? 0,
                    maxSteps: t["max_steps"] as? Int ?? 0)
            }
            await MainActor.run {
                self?.tasks = rows
                self?.rebuildRows()
            }
        }
    }

    // MARK: Agent channel

    /// Sends an `__BADAPPLE_AGENT__` JSON-RPC envelope through the bundled CLI
    /// and returns the daemon's `result` object. Same channel as the Control
    /// Center's agent board.
    private func callAgent(_ method: String, params: [String: Any]) async -> [String: Any]? {
        guard let app = NSApp.delegate as? AppDelegate else { return nil }
        let envelope: [String: Any] = ["id": "tasks-window", "method": method, "params": params]
        guard let data = try? JSONSerialization.data(withJSONObject: envelope),
              let json = String(data: data, encoding: .utf8) else { return nil }
        guard let out = try? await app.runBadAppleCLI(
            prompt: "__BADAPPLE_AGENT__ \(json)",
            socketPath: BadAppleBrain.deepSocket, maxTokens: 4096, timeout: 30) else { return nil }
        guard let resData = out.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: resData) as? [String: Any] else { return nil }
        return obj
    }
}

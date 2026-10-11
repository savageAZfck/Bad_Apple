// BadAppleMenuBarChrome.swift
//
// Menu-bar chrome for Bad Apple: the drawn apple glyph, the live status
// model parsed from the daemon's `runtime status` JSON, the terminal-style
// status header shown at the top of the menu, a transcript panel, and the
// cinematic boot splash. AppKit only, no SwiftUI. Everything here shows
// real status data only — no invented metrics.

import AppKit
import QuartzCore

// MARK: - Palette

enum BadAppleTerm {
    static let panel = NSColor(srgbRed: 0.039, green: 0.039, blue: 0.039, alpha: 1)  // #0A0A0A
    static let splashBG = NSColor(srgbRed: 0.020, green: 0.020, blue: 0.020, alpha: 1)  // #050505
    static let border = NSColor(srgbRed: 0.122, green: 0.122, blue: 0.122, alpha: 1)  // #1F1F1F
    static let green = NSColor(srgbRed: 0.19, green: 0.86, blue: 0.42, alpha: 1)
    static let amber = NSColor(srgbRed: 1.0, green: 0.69, blue: 0.0, alpha: 1)
    static let red = NSColor(srgbRed: 1.0, green: 0.27, blue: 0.23, alpha: 1)
    static let logoRed = NSColor(srgbRed: 1.0, green: 0.231, blue: 0.188, alpha: 1)  // #FF3B30
    static let dim = NSColor(srgbRed: 0.48, green: 0.48, blue: 0.48, alpha: 1)
    static let text = NSColor(srgbRed: 0.82, green: 0.82, blue: 0.82, alpha: 1)

    static func mono(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.monospacedSystemFont(ofSize: size, weight: weight)
    }

    static var versionString: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "dev"
    }

    /// Dim monospace title for the disabled section labels in the menu.
    static func sectionLabel(_ name: String) -> NSAttributedString {
        NSAttributedString(
            string: "── \(name)",
            attributes: [.font: mono(10, .medium), .foregroundColor: dim]
        )
    }
}

// MARK: - Apple glyph

/// The Bad Apple mark: a lopsided rotting apple with a crooked stem, a rot
/// spot cut out of the body, and two drips. Built from Bezier curves in a
/// ~100-unit box (y up). Boolean ops (union/subtract) keep the outline
/// clean for both the template status image and the stroked splash.
enum BadAppleGlyph {
    private static func bodyShape() -> CGPath {
        let p = CGMutablePath()
        p.move(to: CGPoint(x: 50, y: 76))
        p.addCurve(to: CGPoint(x: 90, y: 62), control1: CGPoint(x: 60, y: 87), control2: CGPoint(x: 85, y: 85))
        p.addCurve(to: CGPoint(x: 70, y: 16), control1: CGPoint(x: 96, y: 44), control2: CGPoint(x: 86, y: 23))
        p.addCurve(to: CGPoint(x: 50, y: 17), control1: CGPoint(x: 62, y: 12), control2: CGPoint(x: 56, y: 18))
        p.addCurve(to: CGPoint(x: 24, y: 14), control1: CGPoint(x: 42, y: 16), control2: CGPoint(x: 34, y: 7))
        p.addCurve(to: CGPoint(x: 10, y: 62), control1: CGPoint(x: 8, y: 21), control2: CGPoint(x: 3, y: 42))
        p.addCurve(to: CGPoint(x: 50, y: 76), control1: CGPoint(x: 15, y: 82), control2: CGPoint(x: 40, y: 87))
        p.closeSubpath()
        return p
    }

    private static func stemShape() -> CGPath {
        let p = CGMutablePath()
        p.move(to: CGPoint(x: 44, y: 72))
        p.addLine(to: CGPoint(x: 55, y: 72))
        p.addLine(to: CGPoint(x: 56, y: 84))
        p.addLine(to: CGPoint(x: 65, y: 91))
        p.addLine(to: CGPoint(x: 60, y: 97))
        p.addLine(to: CGPoint(x: 49, y: 90))
        p.addLine(to: CGPoint(x: 45, y: 82))
        p.closeSubpath()
        return p
    }

    private static func rotSpot() -> CGPath {
        let p = CGMutablePath()
        p.move(to: CGPoint(x: 19, y: 37))
        p.addCurve(to: CGPoint(x: 32, y: 24), control1: CGPoint(x: 17, y: 29), control2: CGPoint(x: 25, y: 22))
        p.addCurve(to: CGPoint(x: 44, y: 36), control1: CGPoint(x: 40, y: 26), control2: CGPoint(x: 47, y: 30))
        p.addCurve(to: CGPoint(x: 33, y: 47), control1: CGPoint(x: 43, y: 43), control2: CGPoint(x: 39, y: 46))
        p.addCurve(to: CGPoint(x: 19, y: 37), control1: CGPoint(x: 25, y: 51), control2: CGPoint(x: 20, y: 44))
        p.closeSubpath()
        return p
    }

    /// Teardrop hanging from (cx, top) down `length` units.
    private static func drip(cx: CGFloat, top: CGFloat, length: CGFloat, bulb: CGFloat) -> CGPath {
        let p = CGMutablePath()
        let tip = top - length
        p.move(to: CGPoint(x: cx - 2.8, y: top))
        p.addCurve(
            to: CGPoint(x: cx - bulb, y: tip + bulb),
            control1: CGPoint(x: cx - 2.5, y: top - length * 0.45),
            control2: CGPoint(x: cx - bulb, y: tip + bulb * 2.2))
        p.addArc(center: CGPoint(x: cx, y: tip + bulb), radius: bulb, startAngle: .pi, endAngle: 0, clockwise: false)
        p.addCurve(
            to: CGPoint(x: cx + 2.8, y: top),
            control1: CGPoint(x: cx + bulb, y: tip + bulb * 2.2),
            control2: CGPoint(x: cx + 2.5, y: top - length * 0.45))
        p.closeSubpath()
        return p
    }

    static func longDripPath() -> CGPath { drip(cx: 30, top: 16, length: 28, bulb: 6.5) }
    static func shortDripPath() -> CGPath { drip(cx: 62, top: 19, length: 15, bulb: 5) }

    /// Round droplet that detaches from the long drip tip.
    static func dropletPath() -> CGPath {
        CGPath(ellipseIn: CGRect(x: 30 - 5, y: 16 - 28 - 3, width: 10, height: 10), transform: nil)
    }

    /// Apple body + stem with the rot spot cut out (no drips).
    static func corePath() -> CGPath {
        bodyShape().union(stemShape()).subtracting(rotSpot())
    }

    /// Everything: core + both drips.
    static func fullPath() -> CGPath {
        corePath().union(longDripPath()).union(shortDripPath())
    }

    /// Transform that fits `fullPath()` into `box` (aspect preserved, centered).
    static func transform(fitting box: CGRect) -> CGAffineTransform {
        let b = fullPath().boundingBoxOfPath
        let s = min(box.width / b.width, box.height / b.height)
        return CGAffineTransform(translationX: box.midX - b.midX * s, y: box.midY - b.midY * s)
            .scaledBy(x: s, y: s)
    }

    static func path(fitting box: CGRect) -> CGPath {
        var t = transform(fitting: box)
        return fullPath().copy(using: &t) ?? fullPath()
    }

    static func transformed(_ path: CGPath, fitting box: CGRect) -> CGPath {
        var t = transform(fitting: box)
        return path.copy(using: &t) ?? path
    }

    /// 18×18pt template image for the status item.
    static func statusImage() -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            let path = BadAppleGlyph.path(fitting: rect.insetBy(dx: 1, dy: 0.5))
            guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
            ctx.setFillColor(NSColor.black.cgColor)
            ctx.addPath(path)
            ctx.fillPath(using: .evenOdd)
            return true
        }
        image.isTemplate = true
        return image
    }

    /// The red rotten apple as a full-color logo for welcome/onboarding
    /// windows — same mark as the status item and the splash.
    static func logoView(frame: NSRect) -> NSImageView {
        let image = NSImage(size: frame.size, flipped: false) { rect in
            let path = BadAppleGlyph.path(fitting: rect.insetBy(dx: 2, dy: 2))
            guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
            ctx.setFillColor(NSColor(red: 1.0, green: 0.231, blue: 0.188, alpha: 1).cgColor)
            ctx.addPath(path)
            ctx.fillPath(using: .evenOdd)
            return true
        }
        let view = NSImageView(frame: frame)
        view.image = image
        view.imageScaling = .scaleProportionallyUpOrDown
        return view
    }
}

// MARK: - Status model

/// Everything the header, status dot and splash show, parsed from the
/// daemon's `runtime status` JSON (schema a) or the in-process engine's
/// synthesized dict (schema b).
struct BadAppleStatusModel {
    enum Mode: String {
        case ready = "READY"
        case starting = "STARTING"
        case killed = "KILLED"
        case safe = "SAFE MODE"
        case offline = "OFFLINE"
        case unknown = "UNKNOWN"
    }
    enum Light { case ok, pending, warn, fail, off, na }

    var reachable = false
    var inProcess = false
    var mode: Mode = .unknown
    var killed = false
    var safeReason: String?

    var gpuName: String?
    var gpuLight: Light = .na
    var gpuError: String?

    var aneLight: Light = .na
    var aneSize: String?
    var aneLoadSeconds: Double?
    var aneContext: Int?

    var embedName: String?
    var embedLight: Light = .na

    var autopilot: Bool?
    var privateMode: Bool?

    var memUsedGB: Double?
    var memTotalGB: Double?
    var memPressure: String?

    var queriesOK: Int?
    var queriesFailed: Int?

    var lastError: String?

    var voiceText: String?
    var voiceColor: NSColor = BadAppleTerm.dim

    static func shortName(_ repo: String) -> String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }

    private static func light(_ s: String?) -> Light {
        switch s?.lowercased() {
        case "ready": return .ok
        case "error", "failed": return .fail
        case "loading", "starting": return .pending
        default: return .pending
        }
    }

    private static var manifestSizeCache: [String: String?] = [:]

    /// "4B" parsed from the ANE manifest's source model directory name
    /// (e.g. `qwen3-4b-src`). nil when the manifest carries no such hint.
    static func aneSize(manifestPath: String?) -> String? {
        guard let manifestPath, !manifestPath.isEmpty else { return nil }
        if let cached = manifestSizeCache[manifestPath] { return cached }
        var found: String?
        if let data = FileManager.default.contents(atPath: manifestPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let src = (json["source"] as? [String: Any])?["path"] as? String {
            let leaf = (src as NSString).lastPathComponent
            if let re = try? NSRegularExpression(pattern: "(?<![0-9.])([0-9]+(?:\\.[0-9]+)?)[bB](?![a-zA-Z])"),
               let m = re.firstMatch(in: leaf, range: NSRange(leaf.startIndex..., in: leaf)),
               let r = Range(m.range(at: 1), in: leaf) {
                found = String(leaf[r]).uppercased() + "B"
            }
        }
        manifestSizeCache[manifestPath] = found
        return found
    }

    /// Parse a status dict. `fallbackMem` fills memory when the status
    /// carries none (GB used / total / pressure).
    static func parse(
        status: [String: Any],
        reachable: Bool,
        lastError: String? = nil,
        fallbackMem: (used: Double, total: Double, pressure: String)? = nil,
        fallbackAutopilot: Bool? = nil
    ) -> BadAppleStatusModel {
        var m = BadAppleStatusModel()
        m.reachable = reachable
        m.lastError = (lastError?.isEmpty == false) ? lastError : nil
        guard reachable else {
            m.mode = .offline
            return m
        }

        if status["native_engine"] as? Bool == true || (status["model_status"] == nil && status["model_id"] != nil) {
            // schema (b): in-process engine
            m.inProcess = true
            if let id = status["model_id"] as? String, !id.isEmpty {
                m.gpuName = shortName(id)
            }
            let ready = (status["mode"] as? String)?.uppercased() == "READY"
            m.gpuLight = ready ? .ok : .pending
            m.mode = ready ? .ready : .starting
            if let used = status["memory_gb"] as? Double, used > 0 {
                m.memUsedGB = used
            }
            m.autopilot = fallbackAutopilot
            m.aneLight = .na
        } else {
            // schema (a): daemon
            m.killed = status["killed"] as? Bool ?? false
            if let r = status["safe_mode_reason"] as? String, !r.isEmpty { m.safeReason = r }
            m.autopilot = status["autopilot"] as? Bool ?? fallbackAutopilot
            m.privateMode = status["private_mode"] as? Bool

            let active = status["active_model_ids"] as? [String] ?? []
            let modelStatus = status["model_status"] as? [String: Any] ?? [:]
            func isEmbed(_ repo: String) -> Bool {
                let l = repo.lowercased()
                return l.contains("bge") || l.contains("embed")
            }
            let repos = Array(Set(active + modelStatus.keys)).sorted()
            let mainRepo = repos.first(where: { !isEmbed($0) && active.contains($0) })
                ?? repos.first(where: { !isEmbed($0) })
            let embedRepo = repos.first(where: { isEmbed($0) })
            if let mainRepo {
                m.gpuName = shortName(mainRepo)
                if let st = modelStatus[mainRepo] as? [String: Any] {
                    m.gpuLight = light(st["status"] as? String)
                    if let e = st["error"] as? String, !e.isEmpty {
                        m.gpuError = e
                        m.gpuLight = .fail
                    }
                } else {
                    m.gpuLight = active.contains(mainRepo) ? .ok : .pending
                }
            } else {
                m.gpuLight = .pending
            }
            if let embedRepo {
                m.embedName = shortName(embedRepo)
                if let st = modelStatus[embedRepo] as? [String: Any] {
                    m.embedLight = light(st["status"] as? String)
                    if let e = st["error"] as? String, !e.isEmpty { m.embedLight = .fail }
                } else {
                    m.embedLight = active.contains(embedRepo) ? .ok : .pending
                }
            }

            if let ane = status["ane_brain"] as? [String: Any] {
                let enabled = ane["enabled"] as? Bool ?? true
                let state = (ane["state"] as? String)?.lowercased() ?? ""
                if !enabled || state == "disabled" {
                    m.aneLight = .off
                } else if state == "ready" {
                    m.aneLight = .ok
                } else if state == "failed" || state == "error" {
                    m.aneLight = .fail
                } else {
                    m.aneLight = .pending
                }
                m.aneLoadSeconds = ane["load_seconds"] as? Double
                m.aneContext = ane["context"] as? Int
                m.aneSize = aneSize(manifestPath: ane["manifest"] as? String)
            }

            if let q = status["query_counts"] as? [String: Any] {
                m.queriesOK = q["succeeded"] as? Int
                m.queriesFailed = q["failed"] as? Int
            }
            if let mem = status["memory"] as? [String: Any] {
                if let u = mem["used_bytes"] as? Double { m.memUsedGB = u / 1_073_741_824 }
                if let t = mem["total_bytes"] as? Double { m.memTotalGB = t / 1_073_741_824 }
                m.memPressure = mem["pressure"] as? String
            }

            let sup = (status["supervisor_mode"] as? String)?.uppercased() ?? ""
            if m.killed {
                m.mode = .killed
            } else if m.safeReason != nil || sup == "SAFE_MODE" {
                m.mode = .safe
            } else if m.gpuLight == .ok && (sup.isEmpty || sup == "READY") {
                m.mode = .ready
            } else if sup == "READY" || sup == "STARTING" || m.gpuLight == .pending || m.gpuLight == .na {
                m.mode = .starting
            } else {
                m.mode = .unknown
            }
        }

        if m.memUsedGB == nil, let f = fallbackMem, f.total > 0 {
            m.memUsedGB = f.used
            m.memTotalGB = f.total
            m.memPressure = f.pressure
        }
        return m
    }

    /// Status-item dot: green = reachable + GPU brain ready + not killed;
    /// amber = starting / safe mode / killed; red = unreachable; gray = unknown.
    var dotColor: NSColor {
        if !reachable { return BadAppleTerm.red }
        switch mode {
        case .ready where gpuLight == .ok: return BadAppleTerm.green
        case .ready, .starting, .safe, .killed: return BadAppleTerm.amber
        case .offline: return BadAppleTerm.red
        case .unknown: return .systemGray
        }
    }

    // ---- header rows ----

    struct Row {
        var dot: NSColor?
        var label: String
        var text: String
        var textColor: NSColor
        var value: String
        var valueColor: NSColor
    }

    static func color(_ l: Light) -> NSColor {
        switch l {
        case .ok: return BadAppleTerm.green
        case .pending, .warn: return BadAppleTerm.amber
        case .fail: return BadAppleTerm.red
        case .off, .na: return BadAppleTerm.dim
        }
    }

    var aneDetail: String {
        [aneSize, "neural engine"].compactMap { $0 }.joined(separator: " · ")
    }

    var aneValue: String {
        switch aneLight {
        case .ok:
            if let s = aneLoadSeconds { return String(format: "ready (%.1fs)", s) }
            return "ready"
        case .pending, .warn: return "loading"
        case .fail: return "failed"
        case .off: return "off"
        case .na: return "n/a"
        }
    }

    var rows: [Row] {
        var r: [Row] = []
        let gpuVal: String
        switch gpuLight {
        case .ok: gpuVal = "ready"
        case .pending, .warn: gpuVal = reachable ? "loading" : "offline"
        case .fail: gpuVal = "error"
        case .off, .na: gpuVal = reachable ? "--" : "offline"
        }
        r.append(Row(
            dot: Self.color(reachable ? gpuLight : .fail), label: "gpu",
            text: (gpuName.map { "\($0) · mlx" }) ?? (reachable ? "waiting for model" : "daemon unreachable"),
            textColor: BadAppleTerm.text, value: gpuVal, valueColor: Self.color(reachable ? gpuLight : .fail)))
        r.append(Row(
            dot: Self.color(aneLight), label: "ane", text: aneDetail, textColor: BadAppleTerm.text,
            value: aneValue, valueColor: Self.color(aneLight)))

        let modeColor: NSColor
        switch mode {
        case .ready: modeColor = BadAppleTerm.green
        case .starting, .safe, .killed: modeColor = mode == .killed ? BadAppleTerm.red : BadAppleTerm.amber
        case .offline: modeColor = BadAppleTerm.red
        case .unknown: modeColor = BadAppleTerm.dim
        }
        func onOff(_ v: Bool?) -> String { v.map { $0 ? "ON" : "OFF" } ?? "--" }
        r.append(Row(
            dot: modeColor, label: "mode", text: mode.rawValue, textColor: modeColor,
            value: "autopilot \(onOff(autopilot)) · private \(onOff(privateMode))", valueColor: BadAppleTerm.dim))

        if let v = voiceText {
            r.append(Row(dot: voiceColor, label: "voice", text: v, textColor: voiceColor == BadAppleTerm.dim ? BadAppleTerm.dim : BadAppleTerm.text, value: "", valueColor: .clear))
        }

        var memText = "calibrating..."
        if let u = memUsedGB {
            if let t = memTotalGB, t > 0 {
                memText = String(format: "%.1f / %.1f GB", u, t)
            } else {
                memText = String(format: "%.1f GB", u)
            }
            if let p = memPressure { memText += " · \(p)" }
        }
        let memColor: NSColor = {
            switch memPressure?.lowercased() {
            case "critical": return BadAppleTerm.red
            case "warning", "warn", "elevated": return BadAppleTerm.amber
            case nil: return BadAppleTerm.dim
            default: return BadAppleTerm.green
            }
        }()
        r.append(Row(dot: memColor, label: "mem", text: memText, textColor: BadAppleTerm.text, value: "", valueColor: .clear))

        if let ok = queriesOK {
            let failed = queriesFailed ?? 0
            r.append(Row(
                dot: failed > 0 ? BadAppleTerm.amber : BadAppleTerm.green, label: "q",
                text: "\(ok) ok / \(failed) failed", textColor: BadAppleTerm.text, value: "", valueColor: .clear))
        }
        if let e = lastError, !e.isEmpty {
            r.append(Row(
                dot: BadAppleTerm.red, label: "err", text: "last error: " + String(e.prefix(46)),
                textColor: BadAppleTerm.red, value: "", valueColor: .clear))
        }
        return r
    }
}

// MARK: - Menu header view

/// Terminal-look status panel hosted in the first menu item. `update`
/// mutates the rows in place, so the header stays live even while the
/// menu is open (item removal is unsafe then; redrawing a view is not).
final class BadAppleMenuHeaderView: NSView {
    private var rows: [BadAppleStatusModel.Row] = []
    private let titleHeight: CGFloat = 26
    private let rowHeight: CGFloat = 17
    static let width: CGFloat = 360

    override var isFlipped: Bool { true }

    init() {
        super.init(frame: NSRect(x: 0, y: 0, width: Self.width, height: 100))
        applyHeight()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not supported") }

    private func applyHeight() {
        let h = titleHeight + CGFloat(max(rows.count, 5)) * rowHeight + 12
        if abs(frame.height - h) > 0.5 {
            setFrameSize(NSSize(width: Self.width, height: h))
        }
    }

    func update(_ model: BadAppleStatusModel) {
        rows = model.rows
        applyHeight()
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        BadAppleTerm.panel.setFill()
        bounds.fill()

        let title: [NSAttributedString.Key: Any] = [
            .font: BadAppleTerm.mono(11, .bold), .foregroundColor: BadAppleTerm.green,
        ]
        NSAttributedString(string: "BAD APPLE // sovereign AGI", attributes: title).draw(at: NSPoint(x: 14, y: 6))
        let ver = NSAttributedString(
            string: "v" + BadAppleTerm.versionString,
            attributes: [.font: BadAppleTerm.mono(11), .foregroundColor: BadAppleTerm.dim])
        ver.draw(at: NSPoint(x: bounds.width - 14 - ver.size().width, y: 6))
        BadAppleTerm.border.setFill()
        NSRect(x: 14, y: 22, width: bounds.width - 28, height: 1).fill()

        let font = BadAppleTerm.mono(11)
        var y = titleHeight
        for row in rows {
            if let dot = row.dot {
                NSAttributedString(string: "●", attributes: [.font: BadAppleTerm.mono(9), .foregroundColor: dot])
                    .draw(at: NSPoint(x: 14, y: y + 2))
            }
            NSAttributedString(string: row.label, attributes: [.font: font, .foregroundColor: BadAppleTerm.dim])
                .draw(at: NSPoint(x: 30, y: y))
            NSAttributedString(string: row.text, attributes: [.font: font, .foregroundColor: row.textColor])
                .draw(at: NSPoint(x: 68, y: y))
            if !row.value.isEmpty {
                let v = NSAttributedString(string: row.value, attributes: [.font: font, .foregroundColor: row.valueColor])
                v.draw(at: NSPoint(x: bounds.width - 14 - v.size().width, y: y))
            }
            y += rowHeight
        }
    }
}

// MARK: - Transcript panel

enum BadAppleTranscript {
    /// Modal alert with a scrollable monospaced transcript.
    @MainActor
    static func show(title: String, text: String) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = title
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 600, height: 340))
        scroll.hasVerticalScroller = true
        scroll.borderType = .lineBorder
        let tv = NSTextView(frame: scroll.bounds)
        tv.isEditable = false
        tv.isRichText = false
        tv.font = BadAppleTerm.mono(11)
        tv.textColor = BadAppleTerm.text
        tv.backgroundColor = BadAppleTerm.panel
        tv.textContainerInset = NSSize(width: 8, height: 8)
        tv.isVerticallyResizable = true
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = true
        tv.string = text.isEmpty ? "(no output)" : text
        scroll.documentView = tv
        alert.accessoryView = scroll
        alert.addButton(withTitle: "Close")
        alert.addButton(withTitle: "Copy")
        if alert.runModal() == .alertSecondButtonReturn {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        }
    }
}

// MARK: - Cinematic boot splash

final class BadAppleSplashWindow {
    private struct BootLine {
        let name: String
        var detail: String
        var light: BadAppleStatusModel.Light
        var nameField: NSTextField
        var detailField: NSTextField
        var statusField: NSTextField
        var revealAt: Double
    }

    private var window: NSWindow?
    private var root: NSView?
    private var lines: [BootLine] = []
    private var titleField: NSTextField?
    private var subtitleField: NSTextField?
    private var timerField: NSTextField?
    private var awakeField: NSTextField?
    private var start = Date()
    private var tick: Timer?
    private var daemonOffline = true
    private var finishing = false
    private var finishStartedAt: Double?
    private var gpuOKAt: Double?
    private var awakeText = "> she's awake."

    private let titleString = "BAD APPLE"
    private let subtitleString = "sovereign personal AGI · my data · my machine · my agi"
    private let size = NSSize(width: 640, height: 420)
    private let logStart = 1.8
    private let logStep = 0.28
    private let spinner = ["[ | ]", "[ / ]", "[ - ]", "[ \\ ]"]

    private var elapsed: Double { -start.timeIntervalSinceNow }

    func show() {
        close()
        start = Date()
        finishing = false
        finishStartedAt = nil
        gpuOKAt = nil
        daemonOffline = true

        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width, height: size.height)
        let w = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
        w.backgroundColor = .clear
        w.isOpaque = false
        w.hasShadow = true
        w.level = .statusBar
        w.isReleasedWhenClosed = false

        let view = NSView(frame: NSRect(origin: .zero, size: size))
        view.wantsLayer = true
        guard let layer = view.layer else { return }
        layer.backgroundColor = BadAppleTerm.splashBG.cgColor
        layer.cornerRadius = 12
        layer.borderWidth = 1
        layer.borderColor = BadAppleTerm.border.cgColor
        layer.masksToBounds = true

        // scanlines: one hairline replicated every 3pt
        let line = CALayer()
        line.frame = CGRect(x: 0, y: 0, width: size.width, height: 1)
        line.backgroundColor = NSColor.white.withAlphaComponent(0.04).cgColor
        let rep = CAReplicatorLayer()
        rep.frame = CGRect(origin: .zero, size: size)
        rep.instanceCount = Int(size.height / 3)
        rep.instanceTransform = CATransform3DMakeTranslation(0, 3, 0)
        rep.addSublayer(line)
        rep.zPosition = 10
        layer.addSublayer(rep)

        // glyph: stroke on in red, then fill; drips extend; a droplet falls on a loop
        let red = BadAppleTerm.logoRed.withAlphaComponent(0.9).cgColor
        let box = CGRect(x: 0, y: 0, width: 100, height: 118)
        let fit = box.insetBy(dx: 4, dy: 4)
        let shape = CAShapeLayer()
        shape.frame = CGRect(x: 48, y: 262, width: box.width, height: box.height)
        shape.path = BadAppleGlyph.transformed(BadAppleGlyph.corePath(), fitting: fit)
        shape.fillRule = .evenOdd
        shape.strokeColor = BadAppleTerm.logoRed.cgColor
        shape.fillColor = red
        shape.lineWidth = 2
        shape.lineJoin = .round
        layer.addSublayer(shape)
        let now = CACurrentMediaTime()
        let stroke = CABasicAnimation(keyPath: "strokeEnd")
        stroke.fromValue = 0
        stroke.toValue = 1
        stroke.duration = 1.2
        stroke.beginTime = now + 0.05
        stroke.fillMode = .backwards
        stroke.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        shape.add(stroke, forKey: "stroke")
        let fill = CABasicAnimation(keyPath: "fillColor")
        fill.fromValue = BadAppleTerm.logoRed.withAlphaComponent(0).cgColor
        fill.toValue = red
        fill.duration = 0.5
        fill.beginTime = now + 1.1
        fill.fillMode = .backwards
        shape.add(fill, forKey: "fill")

        for dripPath in [BadAppleGlyph.longDripPath(), BadAppleGlyph.shortDripPath()] {
            let p = BadAppleGlyph.transformed(dripPath, fitting: fit)
            let bb = p.boundingBoxOfPath
            var shift = CGAffineTransform(translationX: -bb.minX, y: -bb.minY)
            let drip = CAShapeLayer()
            drip.bounds = CGRect(origin: .zero, size: bb.size)
            drip.anchorPoint = CGPoint(x: 0.5, y: 1)
            drip.position = CGPoint(x: bb.midX, y: bb.maxY)
            drip.path = p.copy(using: &shift)
            drip.fillColor = red
            shape.addSublayer(drip)
            let grow = CABasicAnimation(keyPath: "transform.scale.y")
            grow.fromValue = 0.02
            grow.toValue = 1
            grow.duration = 1.0
            grow.beginTime = now + 1.7
            grow.fillMode = .backwards
            grow.timingFunction = CAMediaTimingFunction(name: .easeIn)
            drip.add(grow, forKey: "grow")
        }

        let droplet = CAShapeLayer()
        droplet.frame = shape.bounds
        droplet.path = BadAppleGlyph.transformed(BadAppleGlyph.dropletPath(), fitting: fit)
        droplet.fillColor = red
        droplet.opacity = 0
        shape.addSublayer(droplet)
        let fall = CAKeyframeAnimation(keyPath: "transform.translation.y")
        fall.values = [0, 0, -30, -30]
        fall.keyTimes = [0, 0.04, 0.44, 1]
        fall.timingFunctions = [
            CAMediaTimingFunction(name: .linear), CAMediaTimingFunction(name: .easeIn),
            CAMediaTimingFunction(name: .linear),
        ]
        let fade = CAKeyframeAnimation(keyPath: "opacity")
        fade.values = [0, 1, 1, 0, 0]
        fade.keyTimes = [0, 0.04, 0.2, 0.44, 1]
        let loop = CAAnimationGroup()
        loop.animations = [fall, fade]
        loop.duration = 2.5
        loop.repeatCount = .infinity
        loop.beginTime = now + 3.0
        droplet.add(loop, forKey: "droplet")

        func label(_ text: String, size s: CGFloat, weight: NSFont.Weight = .regular,
                   color: NSColor, frame: NSRect) -> NSTextField {
            let f = NSTextField(labelWithString: text)
            f.font = BadAppleTerm.mono(s, weight)
            f.textColor = color
            f.frame = frame
            f.lineBreakMode = .byClipping
            view.addSubview(f)
            return f
        }

        titleField = label("", size: 30, weight: .bold, color: .white,
                           frame: NSRect(x: 184, y: 322, width: 420, height: 40))
        subtitleField = label(subtitleString, size: 11, color: BadAppleTerm.dim,
                              frame: NSRect(x: 184, y: 296, width: 440, height: 16))
        subtitleField?.alphaValue = 0
        timerField = label("t+00.0s", size: 11, color: BadAppleTerm.dim,
                           frame: NSRect(x: size.width - 110, y: size.height - 28, width: 96, height: 16))
        timerField?.alignment = .right
        let ver = label("v" + BadAppleTerm.versionString, size: 11, color: BadAppleTerm.dim,
                        frame: NSRect(x: size.width - 110, y: 12, width: 96, height: 16))
        ver.alignment = .right

        // boot log
        let specs: [(String, String)] = [
            ("daemon", "waiting for daemon"),
            ("gpu brain", "waiting for model"),
            ("ane brain", "neural engine"),
            ("embeddings", "waiting for model"),
            ("governance", "waiting for status"),
            ("safety", "waiting for status"),
        ]
        lines = []
        for (i, spec) in specs.enumerated() {
            let y = 222 - CGFloat(i) * 24
            let n = label("", size: 12, color: BadAppleTerm.text, frame: NSRect(x: 48, y: y, width: 120, height: 18))
            let d = label("", size: 12, color: BadAppleTerm.dim, frame: NSRect(x: 172, y: y, width: 330, height: 18))
            let s = label("", size: 12, weight: .medium, color: BadAppleTerm.amber,
                          frame: NSRect(x: 508, y: y, width: 100, height: 18))
            s.alignment = .right
            lines.append(BootLine(
                name: spec.0, detail: spec.1, light: .pending, nameField: n, detailField: d,
                statusField: s, revealAt: logStart + Double(i) * logStep))
        }
        awakeField = label("", size: 12, weight: .bold, color: BadAppleTerm.green,
                           frame: NSRect(x: 48, y: 64, width: 540, height: 18))

        w.contentView = view
        window = w
        root = view
        w.makeKeyAndOrderFront(nil)

        tick?.invalidate()
        tick = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.onTick()
        }
    }

    // MARK: status in

    /// Feed a status dict — daemon schema (`model_status`, `ane_brain`, …)
    /// or the in-process engine's (`model_id`, `mode`).
    func update(status: [String: Any]) {
        guard window != nil else { return }
        let m = BadAppleStatusModel.parse(status: status, reachable: true)
        daemonOffline = false
        set("daemon", detail: m.inProcess ? "in-process engine" : "substrate.sock", light: .ok)
        set("gpu brain", detail: (m.gpuName.map { "\($0) · mlx" }) ?? "waiting for model",
            light: m.gpuName == nil ? .pending : m.gpuLight, error: m.gpuError)
        var aneDetail = m.aneDetail
        switch m.aneLight {
        case .ok: if let s = m.aneLoadSeconds { aneDetail += String(format: " · %.1fs", s) }
        case .off: aneDetail += " · off"
        case .na: aneDetail = "n/a"
        default: break
        }
        set("ane brain", detail: aneDetail, light: m.aneLight)
        set("embeddings", detail: m.embedName ?? "n/a", light: m.embedName == nil ? .na : m.embedLight)
        if m.autopilot != nil || m.privateMode != nil {
            func f(_ v: Bool?) -> String { v.map { $0 ? "ON" : "OFF" } ?? "--" }
            set("governance", detail: "autopilot \(f(m.autopilot)) · private \(f(m.privateMode))", light: .ok)
        }
        if m.killed {
            set("safety", detail: "KILLED", light: .fail)
        } else if let r = m.safeReason {
            set("safety", detail: "safe mode: \(r)", light: .warn)
        } else if !m.inProcess {
            set("safety", detail: "kill switch clear", light: .ok)
        } else {
            set("safety", detail: "n/a", light: .na)
        }
        if m.gpuLight == .ok, gpuOKAt == nil { gpuOKAt = elapsed }
        lastModel = m
    }

    private var lastModel: BadAppleStatusModel?

    func setOffline() {
        guard window != nil else { return }
        daemonOffline = true
        set("daemon", detail: "waiting for daemon", light: .pending)
    }

    private func set(_ name: String, detail: String, light: BadAppleStatusModel.Light, error: String? = nil) {
        guard let i = lines.firstIndex(where: { $0.name == name }) else { return }
        lines[i].detail = (light == .fail && error != nil) ? "error: " + String(error!.prefix(40)) : detail
        lines[i].light = light
    }

    // MARK: animation

    private func typed(_ s: String, progress: Double) -> String {
        String(s.prefix(Int(Double(s.count) * min(max(progress, 0), 1))))
    }

    private func onTick() {
        guard window != nil else { return }
        let t = elapsed
        timerField?.stringValue = String(format: "t+%04.1fs", t)

        titleField?.stringValue = typed(titleString, progress: (t - 1.0) / 1.2)
        subtitleField?.alphaValue = CGFloat(min(max((t - 2.2) / 0.7, 0), 1))

        let spin = spinner[Int(t * 10) % spinner.count]
        var allRevealed = true
        for line in lines {
            let p = (t - line.revealAt) / 0.3
            if p <= 0 { allRevealed = false; continue }
            if p < 1 { allRevealed = false }
            line.nameField.stringValue = typed(line.name, progress: p * 1.6)
            line.detailField.stringValue = typed(line.detail, progress: p)
            line.detailField.textColor = line.light == .fail ? BadAppleTerm.red : BadAppleTerm.dim
            if p >= 1 {
                switch line.light {
                case .pending:
                    line.statusField.stringValue = spin
                    line.statusField.textColor = BadAppleTerm.amber
                case .ok:
                    line.statusField.stringValue = "[ ok ]"
                    line.statusField.textColor = BadAppleTerm.green
                case .warn:
                    line.statusField.stringValue = "[warn]"
                    line.statusField.textColor = BadAppleTerm.amber
                case .fail:
                    line.statusField.stringValue = "[fail]"
                    line.statusField.textColor = BadAppleTerm.red
                case .off, .na:
                    line.statusField.stringValue = "[ -- ]"
                    line.statusField.textColor = BadAppleTerm.dim
                }
            }
        }

        if finishing {
            if let at = finishStartedAt {
                awakeField?.stringValue = typed(awakeText, progress: (t - at) / 0.5)
                if t - at > 0.5 + 1.5 { fadeOutAndClose() }
            }
            return
        }
        if t > 150 { close(); return }
        guard allRevealed, !daemonOffline, let m = lastModel else { return }
        let anePending = m.aneLight == .pending || m.aneLight == .warn
        if m.gpuLight == .ok && !anePending {
            beginFinish(awake: "> she's awake.")
        } else if m.gpuLight == .ok, let ok = gpuOKAt, t - ok > 20 {
            beginFinish(awake: "> she's awake. (ane brain still loading in background)")
        }
    }

    private func beginFinish(awake: String) {
        finishing = true
        awakeText = awake
        finishStartedAt = elapsed
    }

    private func fadeOutAndClose() {
        tick?.invalidate()
        tick = nil
        guard let w = window else { return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.4
            w.animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            self?.close()
        })
    }

    func close() {
        tick?.invalidate()
        tick = nil
        window?.orderOut(nil)
        window = nil
        root = nil
        lines = []
        titleField = nil
        subtitleField = nil
        timerField = nil
        awakeField = nil
        lastModel = nil
    }
}

import Foundation

private func makeInfoScript() -> String {
    return """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set appName to name of p
        set w to front window of p
        set winName to name of w
        set elements to {}
        set counter to 0
        repeat with e in (entire contents of w)
            try
                if counter > 100 then exit repeat
                if exists e then
                    set n to name of e
                    set r to role of e
                    if n is not missing value then
                        set end of elements to (r & ": " & n)
                        set counter to counter + 1
                    end if
                end if
            end try
        end repeat
        return appName & "|" & winName & "|" & (elements as string)
    end tell
    """
}

private func makeClickScript(target: String) -> String {
    let escaped = target.replacingOccurrences(of: "\"", with: "\\\"")
    return """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "\(escaped)" then
                    click e
                    return "clicked \(escaped)"
                end if
            end try
        end repeat
        return "not found"
    end tell
    """
}

private func makeTypeScript(target: String, text: String) -> String {
    let escTarget = target.replacingOccurrences(of: "\"", with: "\\\"")
    let escText = text.replacingOccurrences(of: "\"", with: "\\\"")
    return """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "\(escTarget)" then
                    set value of e to "\(escText)"
                    return "typed into \(escTarget)"
                end if
            end try
        end repeat
        return "not found"
    end tell
    """
}

private func runAppleScript(_ source: String) -> String? {
    var errorInfo: NSDictionary?
    guard let script = NSAppleScript(source: source) else { return nil }
    let result = script.executeAndReturnError(&errorInfo)
    if let error = errorInfo {
        return "ERROR: \(error)"
    }
    return result.stringValue
}

@main
struct BadAppleUI {
    static func main() {
        let args = CommandLine.arguments
        guard let actionIndex = args.firstIndex(of: "--action"), actionIndex + 1 < args.count else {
            print("Usage: BadAppleUI --action <info|click|type> [--target <name>] [--text <text>]")
            exit(1)
        }
        let action = args[actionIndex + 1]
        let target = (args.firstIndex(of: "--target").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) ?? ""
        let text = (args.firstIndex(of: "--text").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) ?? ""

        var source = ""
        switch action {
        case "info":
            source = makeInfoScript()
        case "click":
            source = makeClickScript(target: target)
        case "type":
            source = makeTypeScript(target: target, text: text)
        default:
            print("Unknown action: \(action)")
            exit(1)
        }

        guard let output = runAppleScript(source) else {
            print("Failed to execute AppleScript")
            exit(1)
        }
        print(output)
        if output.hasPrefix("ERROR:") {
            exit(1)
        }
    }
}

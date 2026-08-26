import Foundation

private func json(_ value: Any) -> String {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: .sortedKeys)
        return String(data: data, encoding: .utf8) ?? "{}"
    } catch {
        return "{}"
    }
}

@main
struct BadAppleUI {
    static func main() {
        let args = CommandLine.arguments
        guard let actionIndex = args.firstIndex(of: "--action"), actionIndex + 1 < args.count else {
            print("Usage: BadAppleUI --action <info|click|type|focus> [--target <name>] [--role <role>] [--text <text>]")
            exit(1)
        }
        let action = args[actionIndex + 1]
        let target = (args.firstIndex(of: "--target").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) ?? ""
        let role = (args.firstIndex(of: "--role").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) ?? ""
        let text = (args.firstIndex(of: "--text").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) ?? ""

        let access = BadAppleUIAccess.shared

        switch action {
        case "info":
            do {
                let root = access.runInfo()
                let data = try JSONEncoder().encode(root)
                print(String(data: data, encoding: .utf8) ?? "{}")
                exit(0)
            } catch {
                print(json(["ok": false, "error": "json encode: \(error)"]))
                exit(1)
            }
        case "click":
            print(json(access.runClick(target: target, role: role)))
        case "type":
            print(json(access.runType(target: target, text: text)))
        case "focus":
            print(json(access.runFocus(target: target)))
        default:
            print(json(["ok": false, "error": "unknown action \(action)"]))
            exit(1)
        }
    }
}

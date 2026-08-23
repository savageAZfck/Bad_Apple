import SafariServices

class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {
    func beginRequest(with context: NSExtensionContext) {
        let item = context.inputItems.first as? NSExtensionItem
        let message = item?.userInfo?["message"]
        let response = NSExtensionItem()
        response.userInfo = [NSExtensionItem.userInfoTypeKey: ["echo": message ?? [:]]]
        context.completeRequest(returningItems: [response], completionHandler: nil)
    }
}

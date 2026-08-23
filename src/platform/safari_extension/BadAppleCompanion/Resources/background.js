// Bad Apple Safari Companion — background service worker
// Bridges the extension to the Bad Apple native messaging host.

const NATIVE_HOST = "com.badapple.companion";

function sendToBadApple(message) {
  return new Promise((resolve, reject) => {
    try {
      browser.runtime.sendNativeMessage(NATIVE_HOST, message, (response) => {
        if (browser.runtime.lastError) {
          reject(browser.runtime.lastError.message);
        } else {
          resolve(response);
        }
      });
    } catch (e) {
      reject(e.message);
    }
  });
}

browser.action.onClicked.addListener(async (tab) => {
  try {
    const page = await browser.tabs.sendMessage(tab.id, { action: "getPageContext" });
    const result = await sendToBadApple({
      type: "summarize_page",
      title: page.title,
      url: page.url,
      text: page.text,
    });
    await browser.tabs.sendMessage(tab.id, { action: "showSummary", summary: result });
  } catch (e) {
    console.error("[badapple companion]", e);
  }
});

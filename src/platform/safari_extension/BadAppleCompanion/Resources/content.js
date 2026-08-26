// Bad Apple Safari Companion — content script
// Extracts the current page title, URL, and visible text, then sends it
// to the background worker on request.

(function () {
  "use strict";

  function getPageContext() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const parts = [];
    let node;
    while ((node = walker.nextNode())) {
      const text = node.textContent.trim();
      if (text.length > 0) parts.push(text);
    }
    return {
      title: document.title,
      url: window.location.href,
      text: parts.join(" ").replace(/\s+/g, " ").slice(0, 8000),
    };
  }

  browser.runtime.onMessage.addListener((request, _sender, sendResponse) => {
    if (request && request.action === "getPageContext") {
      sendResponse(getPageContext());
      return true;
    }
    if (request && request.action === "showSummary") {
      const existing = document.getElementById("badapple-local-summary");
      if (existing) existing.remove();
      const panel = document.createElement("aside");
      panel.id = "badapple-local-summary";
      panel.setAttribute("role", "dialog");
      panel.style.cssText = "position:fixed;right:20px;top:20px;z-index:2147483647;width:360px;max-height:70vh;overflow:auto;padding:16px;border-radius:12px;background:#111;color:#fff;box-shadow:0 8px 32px #0008;font:14px -apple-system,sans-serif;white-space:pre-wrap";
      panel.textContent = request.summary?.text || request.summary || "No summary returned.";
      const close = document.createElement("button");
      close.textContent = "Close";
      close.style.cssText = "display:block;margin-top:12px;padding:6px 10px";
      close.addEventListener("click", () => panel.remove());
      panel.appendChild(close);
      document.documentElement.appendChild(panel);
      sendResponse({ shown: true });
      return true;
    }
    return false;
  });
})();

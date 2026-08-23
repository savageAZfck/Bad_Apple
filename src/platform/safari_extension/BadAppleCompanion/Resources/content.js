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
    return false;
  });
})();

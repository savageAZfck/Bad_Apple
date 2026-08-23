// Bad Apple Safari Companion — popup

document.getElementById("ask").addEventListener("click", async () => {
  const status = document.getElementById("status");
  const prompt = document.getElementById("prompt").value.trim();
  status.textContent = "Sending to Bad Apple...";
  try {
    const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
    const page = await browser.tabs.sendMessage(tab.id, { action: "getPageContext" });
    const response = await browser.runtime.sendNativeMessage("com.badapple.companion", {
      type: "query",
      prompt,
      page,
    });
    status.textContent = response.text || "Sent.";
  } catch (e) {
    status.textContent = "Error: " + e.message;
  }
});

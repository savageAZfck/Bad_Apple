(function () {
  const token = document.querySelector('meta[name="csrf-token"]')?.content;
  if (!token) return;

  const originalFetch = window.fetch;
  window.fetch = function (url, init) {
    const options = init || {};
    const method = (options.method || "GET").toUpperCase();
    const sameOrigin = new URL(url, window.location.href).origin === window.location.origin;
    if (sameOrigin && method !== "GET") {
      const headers = new Headers(options.headers || {});
      headers.set("X-CSRF-Token", token);
      options.headers = headers;
    }
    return originalFetch(url, options);
  };
})();

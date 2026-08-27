(function () {
  function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : '';
  }

  window.csrfToken = csrfToken;
  window.getCsrfToken = csrfToken;

  window.refreshCsrfToken = async function () {
    try {
      const r = await fetch('/api/csrf', { method: 'GET', cache: 'no-store' });
      if (!r.ok) return false;
      const data = await r.json();
      const meta = document.querySelector('meta[name="csrf-token"]');
      if (meta && data.csrf_token) {
        meta.content = data.csrf_token;
        return true;
      }
    } catch (e) {
      console.warn('refreshCsrfToken failed:', e);
    }
    return false;
  };

  const originalFetch = window.fetch;
  window.fetch = function (url, init) {
    const options = init || {};
    const method = (options.method || 'GET').toUpperCase();
    const sameOrigin = new URL(url, window.location.href).origin === window.location.origin;
    if (sameOrigin && method !== 'GET' && method !== 'HEAD') {
      const headers = new Headers(options.headers || {});
      const token = csrfToken();
      if (token && !headers.get('X-CSRF-Token')) {
        headers.set('X-CSRF-Token', token);
      }
      options.headers = headers;
    }
    return originalFetch(url, options);
  };

  document.addEventListener('DOMContentLoaded', () => {
    if (window.refreshCsrfToken) refreshCsrfToken();
  });
})();

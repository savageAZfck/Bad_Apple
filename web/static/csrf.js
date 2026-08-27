(function () {
  function cookieToken() {
    const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : '';
  }

  function metaToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : '';
  }

  function csrfToken() {
    return cookieToken() || metaToken();
  }

  window.csrfToken = csrfToken;
  window.getCsrfToken = csrfToken;

  function setMetaToken(token) {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta && token) meta.content = token;
  }

  window.refreshCsrfToken = async function () {
    try {
      const r = await fetch('/api/csrf', { method: 'GET', cache: 'no-store' });
      if (!r.ok) return false;
      const data = await r.json();
      if (data.csrf_token) {
        setMetaToken(data.csrf_token);
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

  function setupMobileNav() {
    if (document.querySelector('.mobile-header')) return;
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    const header = document.createElement('header');
    header.className = 'mobile-header';
    header.innerHTML = '<button class="menu-btn" aria-label="Open navigation">☰</button>';
    document.body.insertBefore(header, document.body.firstChild);
    const btn = header.querySelector('.menu-btn');
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
    });
    sidebar.querySelectorAll('a').forEach(a => a.addEventListener('click', () => sidebar.classList.remove('open')));
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !header.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (window.refreshCsrfToken) refreshCsrfToken();
    setupMobileNav();
  });
})();

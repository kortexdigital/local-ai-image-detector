/**
 * Finding images on the page and showing what the detector thinks of them.
 *
 * Two rules shape everything here. Analysis must be lazy, because a page can
 * hold hundreds of images and only a handful are ever looked at; and the
 * overlay must never change how the page behaves, because a detector that
 * breaks the sites it runs on is worse than no detector.
 */

const SETTINGS_DEFAULTS = {
  enabled: true,
  threshold: 0.65,
  minImageSide: 128,
  showRealBadges: true,
};

let settings = { ...SETTINGS_DEFAULTS };

/** img -> {badge, result, state} */
const tracked = new Map();
const queued = new WeakSet();

/* ------------------------------ badge ------------------------------ */

function createBadge() {
  const badge = document.createElement('div');
  badge.className = 'aiid-badge aiid-pending';
  badge.textContent = '...';
  badge.setAttribute('role', 'status');
  document.documentElement.appendChild(badge);
  return badge;
}

function positionBadge(badge, image) {
  const rect = image.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) {
    badge.style.display = 'none';
    return;
  }
  badge.style.display = '';
  badge.style.top = `${rect.top + window.scrollY + 6}px`;
  badge.style.left = `${rect.left + window.scrollX + 6}px`;
}

function renderBadge(badge, result) {
  const percent = Math.round(result.confidence * 100);
  const isAi = result.verdict === 'ai';

  badge.classList.remove('aiid-pending', 'aiid-ai', 'aiid-real', 'aiid-error');
  badge.classList.add(isAi ? 'aiid-ai' : 'aiid-real');
  badge.textContent = isAi ? `AI ${percent}%` : `Real ${100 - percent}%`;

  const lines = [
    `AI-generated confidence: ${(result.confidence * 100).toFixed(1)}%`,
    `Flagged at or above ${(settings.threshold * 100).toFixed(0)}%`,
    `Backend: ${result.backend}`,
  ];
  if (result.reason) lines.push(`Signal: ${result.reason}`);
  badge.title = lines.join('\n');
}

function renderError(badge, message) {
  badge.classList.remove('aiid-pending', 'aiid-ai', 'aiid-real');
  badge.classList.add('aiid-error');
  badge.textContent = '?';
  badge.title = `Could not analyze this image: ${message}`;
}

/* ---------------------------- eligibility ---------------------------- */

function isEligible(image) {
  if (!settings.enabled) return false;
  if (!image.currentSrc && !image.src) return false;

  const source = image.currentSrc || image.src;
  // A data: or blob: URL is already local; both are fine to fetch. Anything
  // that is not http(s), data or blob cannot be fetched back.
  if (!/^(https?:|data:|blob:|file:)/.test(source)) return false;

  const naturalSide = Math.min(image.naturalWidth || 0, image.naturalHeight || 0);
  const renderedSide = Math.min(image.clientWidth, image.clientHeight);
  const side = Math.max(naturalSide, renderedSide);
  return side >= settings.minImageSide;
}

/* ------------------------------ analysis ------------------------------ */

async function analyze(image) {
  if (queued.has(image)) return;
  queued.add(image);

  const source = image.currentSrc || image.src;
  let entry = tracked.get(image);
  if (!entry) {
    entry = { badge: createBadge(), result: null };
    tracked.set(image, entry);
  }
  positionBadge(entry.badge, image);

  try {
    const response = await chrome.runtime.sendMessage({ type: 'analyze', url: source });
    if (!response?.ok) throw new Error(response?.error ?? 'analysis failed');

    entry.result = response;
    if (response.verdict === 'real' && !settings.showRealBadges) {
      entry.badge.remove();
      tracked.delete(image);
      return;
    }
    renderBadge(entry.badge, response);
    positionBadge(entry.badge, image);
  } catch (error) {
    renderError(entry.badge, String(error?.message ?? error));
  }
}

/* ------------------------------ observers ------------------------------ */

const visibility = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const image = entry.target;
      if (isEligible(image)) analyze(image);
      visibility.unobserve(image);
    }
  },
  { rootMargin: '200px' },
);

function consider(image) {
  if (tracked.has(image) || queued.has(image)) return;
  if (image.complete && image.naturalWidth > 0) {
    if (isEligible(image)) visibility.observe(image);
  } else {
    image.addEventListener(
      'load',
      () => {
        if (isEligible(image)) visibility.observe(image);
      },
      { once: true },
    );
  }
}

function scan(root = document) {
  for (const image of root.querySelectorAll('img')) consider(image);
}

const mutations = new MutationObserver((records) => {
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      if (node.tagName === 'IMG') consider(node);
      else scan(node);
    }
  }
});

/* ---------------------------- repositioning ---------------------------- */

let repositionScheduled = false;
function scheduleReposition() {
  if (repositionScheduled) return;
  repositionScheduled = true;
  requestAnimationFrame(() => {
    repositionScheduled = false;
    for (const [image, entry] of tracked) {
      if (!image.isConnected) {
        entry.badge.remove();
        tracked.delete(image);
        continue;
      }
      positionBadge(entry.badge, image);
    }
  });
}

/* ------------------------------- startup ------------------------------- */

async function start() {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'getSettings' });
    if (response?.ok) settings = { ...SETTINGS_DEFAULTS, ...response.settings };
  } catch {
    // Keep the defaults if the service worker is still waking up.
  }
  if (!settings.enabled) return;

  scan();
  mutations.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener('scroll', scheduleReposition, { passive: true });
  window.addEventListener('resize', scheduleReposition, { passive: true });
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'sync') return;
  for (const [key, { newValue }] of Object.entries(changes)) {
    settings[key] = newValue;
  }
});

start();

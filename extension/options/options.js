import { DEFAULT_SETTINGS } from '../shared/constants.js';

const fields = {
  enabled: document.getElementById('enabled'),
  threshold: document.getElementById('threshold'),
  minImageSide: document.getElementById('minImageSide'),
  showRealBadges: document.getElementById('showRealBadges'),
};
const saved = document.getElementById('saved');

function announce() {
  saved.textContent = 'Saved.';
  setTimeout(() => {
    saved.textContent = '';
  }, 1200);
}

async function load() {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  const settings = { ...DEFAULT_SETTINGS, ...stored };
  fields.enabled.checked = settings.enabled;
  fields.threshold.value = Math.round(settings.threshold * 100);
  fields.minImageSide.value = settings.minImageSide;
  fields.showRealBadges.checked = settings.showRealBadges;
}

async function save() {
  const percent = Number(fields.threshold.value);
  const side = Number(fields.minImageSide.value);
  await chrome.storage.sync.set({
    enabled: fields.enabled.checked,
    // Clamped so a stray keystroke cannot produce a threshold of 0 or 1,
    // which would classify every image the same way.
    threshold: Math.min(0.99, Math.max(0.01, (Number.isFinite(percent) ? percent : 65) / 100)),
    minImageSide: Math.min(1024, Math.max(16, Number.isFinite(side) ? side : 128)),
    showRealBadges: fields.showRealBadges.checked,
  });
  announce();
}

for (const field of Object.values(fields)) field.addEventListener('change', save);
load();

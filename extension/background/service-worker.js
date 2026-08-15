/**
 * Orchestration.
 *
 * The service worker owns what the content script cannot: the single
 * offscreen inference host, the work queue that keeps a gallery page from
 * saturating the machine, and the per-tab result cache the popup reads.
 *
 * It deliberately does not touch image bytes. chrome.runtime.sendMessage
 * serializes as JSON, so an ArrayBuffer does not survive the trip to the
 * offscreen document; the offscreen document fetches each image itself,
 * running at the extension origin with the same host permissions.
 */

import { DEFAULT_SETTINGS, MAX_CONCURRENT } from '../shared/constants.js';

const OFFSCREEN_PATH = 'offscreen/offscreen.html';
const CACHE_LIMIT = 500;

/** url -> result, so re-scrolling or revisiting never recomputes. */
const resultCache = new Map();

/** tabId -> Map(url -> result), for the popup. */
const perTab = new Map();

let queueDepth = 0;
const pending = [];
let creatingOffscreen = null;

async function hasOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreenDocument()) return;
  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }
  creatingOffscreen = chrome.offscreen
    .createDocument({
      url: OFFSCREEN_PATH,
      // BLOBS covers decoding image data with createImageBitmap and
      // OffscreenCanvas, which is what the host actually does.
      reasons: ['BLOBS'],
      justification:
        'Runs the local image classifier once per browser instead of once per tab.',
    })
    .catch((error) => {
      // A concurrent call may have created it between the check and this call.
      if (!String(error).includes('Only a single offscreen')) throw error;
    })
    .finally(() => {
      creatingOffscreen = null;
    });
  await creatingOffscreen;
}

function cacheResult(url, result) {
  if (resultCache.size >= CACHE_LIMIT) {
    resultCache.delete(resultCache.keys().next().value);
  }
  resultCache.set(url, result);
}

async function getSettings() {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function analyze(url, tabId) {
  const cached = resultCache.get(url);
  if (cached) {
    rememberForTab(tabId, cached);
    return cached;
  }

  const settings = await getSettings();
  await ensureOffscreen();

  const result = await chrome.runtime.sendMessage({
    target: 'offscreen',
    type: 'scoreUrl',
    url,
    threshold: settings.threshold,
  });
  if (!result?.ok) throw new Error(result?.error ?? 'inference failed');

  const { ok, ...payload } = result;
  cacheResult(url, payload);
  rememberForTab(tabId, payload);
  return payload;
}

function rememberForTab(tabId, result) {
  if (tabId == null) return;
  if (!perTab.has(tabId)) perTab.set(tabId, new Map());
  perTab.get(tabId).set(result.url, result);
}

/** Serialize work so a gallery page does not spawn a hundred decodes at once. */
function enqueue(task) {
  return new Promise((resolve, reject) => {
    pending.push({ task, resolve, reject });
    drain();
  });
}

function drain() {
  while (queueDepth < MAX_CONCURRENT && pending.length > 0) {
    const { task, resolve, reject } = pending.shift();
    queueDepth += 1;
    task()
      .then(resolve, reject)
      .finally(() => {
        queueDepth -= 1;
        drain();
      });
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Messages addressed to the offscreen document are not ours to handle.
  if (message?.target === 'offscreen') return false;

  if (message?.type === 'analyze') {
    const tabId = sender.tab?.id;
    enqueue(() => analyze(message.url, tabId))
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) =>
        sendResponse({ ok: false, error: String(error?.message ?? error) }),
      );
    return true;
  }

  if (message?.type === 'getTabResults') {
    const results = perTab.has(message.tabId)
      ? [...perTab.get(message.tabId).values()]
      : [];
    sendResponse({ ok: true, results });
    return false;
  }

  if (message?.type === 'getSettings') {
    getSettings().then((settings) => sendResponse({ ok: true, settings }));
    return true;
  }

  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => perTab.delete(tabId));
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'loading') perTab.delete(tabId);
});

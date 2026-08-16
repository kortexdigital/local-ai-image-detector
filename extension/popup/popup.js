/** Summary of what the detector found on the active tab. */

const statusEl = document.getElementById('status');
const listEl = document.getElementById('results');
const backendEl = document.getElementById('backend');

document.getElementById('open-options').addEventListener('click', (event) => {
  event.preventDefault();
  chrome.runtime.openOptionsPage();
});

function row(result) {
  const li = document.createElement('li');

  const thumb = document.createElement('img');
  thumb.src = result.url;
  thumb.alt = '';
  thumb.loading = 'lazy';

  const meta = document.createElement('div');
  meta.className = 'meta';
  const name = document.createElement('div');
  name.className = 'name';
  try {
    const url = new URL(result.url);
    name.textContent = url.pathname.split('/').pop() || url.hostname;
  } catch {
    name.textContent = 'image';
  }
  meta.appendChild(name);

  if (result.reason) {
    const reason = document.createElement('div');
    reason.className = 'reason';
    reason.textContent = result.reason;
    meta.appendChild(reason);
  }

  const score = document.createElement('div');
  score.className = `score ${result.verdict}`;
  score.textContent = `${Math.round(result.confidence * 100)}%`;
  score.title = 'Probability this image is AI-generated';

  li.append(thumb, meta, score);
  return li;
}

async function render() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    statusEl.textContent = 'No active tab.';
    return;
  }

  const response = await chrome.runtime.sendMessage({
    type: 'getTabResults',
    tabId: tab.id,
  });
  const results = response?.results ?? [];

  if (results.length === 0) {
    statusEl.textContent = 'No images analyzed yet. Scroll the page to analyze more.';
    return;
  }

  // Most suspicious first: that is the reason someone opens this popup.
  results.sort((a, b) => b.confidence - a.confidence);

  const flagged = results.filter((r) => r.verdict === 'ai').length;
  statusEl.textContent = `${results.length} analyzed, ${flagged} flagged as AI-generated.`;
  backendEl.textContent = `Inference: ${results[0].backend} (on-device)`;

  for (const result of results) listEl.appendChild(row(result));
}

render().catch((error) => {
  statusEl.textContent = `Could not load results: ${error.message}`;
});

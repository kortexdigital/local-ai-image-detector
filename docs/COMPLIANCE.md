# Compliance

Each requirement of the challenge, mapped to the mechanism that satisfies it
and to how anyone can check it independently.

## Requirements

| Requirement | How it is met | How to verify |
|---|---|---|
| MIT licensed | `LICENSE` at the repository root; copied into `dist/` by the build | `cat LICENSE` |
| Native Manifest V3 Chrome extension | `extension/manifest.json` declares `manifest_version: 3`, a service worker, and an MV3 content security policy | `eval/tests/build.test.mjs` asserts it; load `dist/` in `chrome://extensions` |
| All inference local, using browser technologies | ONNX Runtime Web running WebGPU when available and WebAssembly otherwise, inside an offscreen document | `extension/offscreen/offscreen.js`; the harness prints the backend it ran on |
| One-time model download during setup only | The build downloads the backbone once, to a pinned SHA-256, and bundles it into `dist/`. The extension performs no download at any point | `scripts/build.mjs`; run with the network off after building |
| Automatic analysis of images on ordinary pages | Content script with `MutationObserver` and `IntersectionObserver` over every `img` | `extension/content/content.js`; open any page with images |
| A confidence score for every analyzed image | A badge per image, plus a list in the popup | `extension/content/content.js`, `extension/popup/popup.js` |
| Complete build and installation instructions | `README.md` sections *Build* and *Install* | Follow them from a clean clone |
| Fully reproducible from source | `npm ci` locks every JavaScript dependency; the backbone is pinned by SHA-256; the training pipeline is included with fixed seeds | `npm ci && npm run build` twice produces identical `dist/models/` |

## Rules

| Rule | Status | Mechanism |
|---|---|---|
| No cloud inference | Satisfied | Inference is three local ONNX graphs. There is no inference endpoint anywhere in the codebase |
| No image data sent to external services | Satisfied | The extension makes exactly one kind of network request: fetching the bytes of an image already loaded on the page, from the browser cache. Nothing is uploaded |
| No local Python, Node, Flask or similar backend | Satisfied | `dist/` contains only the extension. Node appears in `scripts/` (build) and `eval/` (measurement), and Python in `training/`, none of which ship or run at browsing time |
| No models downloaded after initial setup | Satisfied | All three graphs and the calibration are bundled into `dist/` at build time. The runtime loads them with `chrome.runtime.getURL`, never over the network |
| No hardcoded benchmark hashes or lookup tables | Satisfied | No code path is keyed on image identity. The only hash in the project is the SHA-256 of the backbone file, used at build time to verify the download |
| No circumvention of the evaluation process | Satisfied | The harness drives the shipped extension through the same function the browsing path uses, so what is measured is what is installed |

## Verifying the offline claim

The strongest check is behavioural:

1. `npm ci && npm run build`
2. Load `dist/` unpacked in a fresh Chrome profile
3. Visit a page with images and confirm badges appear
4. Disconnect the network, or set DevTools to Offline
5. Reload and confirm scoring still happens

Static checks:

- `eval/tests/build.test.mjs` fails if any shipped script fetches an `http` or
  `https` URL.
- `ort.env.wasm.wasmPaths` is set to a `chrome-extension://` URL in
  `extension/offscreen/offscreen.js`, so the runtime resolves its WebAssembly
  from the package rather than from a CDN, which is the default and the usual
  source of an accidental network dependency.
- The manifest's content security policy is `script-src 'self'
  'wasm-unsafe-eval'`, which blocks remote script outright.

## Notes on scope

The build and the evaluation harness require Node. That is tooling, not a
runtime dependency: the maintainer runs the build once to produce `dist/`, and
the extension inside `dist/` has no dependency on Node, Python, localhost, or
anything else outside the browser.

The training pipeline is included so the model can be reproduced and audited
rather than taken on trust. It never runs during browsing.

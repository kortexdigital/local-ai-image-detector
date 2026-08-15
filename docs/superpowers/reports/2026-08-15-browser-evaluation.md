# Browser evaluation

**Date:** 2026-08-15
**Measured through:** the built extension, driven in a real Chrome by
`eval/run-benchmark.mjs`

## Headline

| Metric | Value |
|---|---|
| **Balanced accuracy at the 0.65 threshold** | **0.8273** |
| Required to qualify | 0.7500 |
| True positive rate (AI detected) | 0.7931 |
| True negative rate (real kept) | 0.8615 |
| Images scored | 2600 |
| Backend | WebAssembly (headless Chrome exposes no WebGPU adapter) |
| Throughput | 2.88 images per second |

Confusion matrix at 0.65:

| | predicted real | predicted AI |
|---|---|---|
| **actually real** | 1120 | 180 |
| **actually AI** | 269 | 1031 |

Per generator, none of which appeared in training:

| Generator | Accuracy | |
|---|---|---|
| nano-banana | 0.9900 | Gemini image model, 2025 |
| real | 0.8615 | |
| flux | 0.8417 | rectified-flow transformer |
| biggan | 0.4333 | GAN, the known weak point |

The balanced accuracy curve peaks at 0.8304 at a threshold of 0.60, so the
calibration placed the decision boundary within one grid step of the optimum.

## What was measured, and on what

The 2600 images are the reporting half of the held-out split: generators the
head never saw during training, and real images it never saw either. The
calibration threshold was fitted on the *other* half, so the threshold never
saw the images it is reported against.

The harness loads `dist/` into Chrome and calls the same scoring function the
browsing path calls. The number describes the extension as installed, not a
reimplementation of it.

## Quantization

The shipped backbone is the int8 export, 89 MB against 350 MB for fp32. The
head was trained on features from that same quantized graph rather than on
fp32 features, so the browser runs the model that was fitted rather than an
approximation of one.

| Configuration | Balanced accuracy |
|---|---|
| fp32 backbone, measured in Python | 0.8362 |
| int8 backbone, measured in Python | 0.8212 |
| int8 backbone, measured in the browser | 0.8273 |

Quantization costs about 1.5 points and saves 260 MB. With seven points of
margin over the bar, that is a good trade.

## Preprocessing parity

`eval/parity.mjs` compares the browser against the Python pipeline stage by
stage:

| Stage | Agreement | |
|---|---|---|
| Decode and centre crop | exact, pixel sums identical | |
| Preprocessing graph | 7.15e-7 maximum absolute difference | float32 rounding |
| Backbone embedding | 0.983 cosine similarity | int8 kernels differ across execution providers |

The first two are exact because both sides run the same ONNX graph. That was
the point of putting resize and normalization inside a graph instead of
writing them twice.

The third cannot be exact. onnxruntime's native CPU build and its WebAssembly
build requantize int8 differently. This was worth chasing down rather than
assuming, and the answer is that it costs nothing measurable: the browser
figure came out slightly *above* the Python one on identical images.

Notably, the browser and Python agreeing bit-for-bit on decode was not
automatic. `createImageBitmap` applies an embedded ICC profile by default
while the training pipeline reads raw samples, so the decode call passes
`colorSpaceConversion: 'none'`.

## End to end

`eval/smoke.mjs` loads the built extension into Chrome, serves a page of six
images, scrolls it as a user would, and asserts that every image gets a badge
carrying a percentage and a verdict, and that the overlay does not intercept
clicks. It passes.

## Offline

`eval/offline-check.mjs` puts the browser offline before the extension loads
anything and aborts every request that is not a `chrome-extension:` URL. The
models load, both a real and an AI image score correctly, and the count of
blocked requests is zero: the extension made no non-extension request at all.

## Reproducibility

Building from a fresh clone produces a byte-identical `dist/`: the same
aggregate hash over every text asset, the same backbone digest, the same head
digest.

## Remaining weakness

BigGAN sits at 0.43 while both modern held-out generators clear 0.84. GAN
artifacts do not transfer from the GAN families available for training, which
are all face generators, to BigGAN's ImageNet-style objects. It is 300 of the
1300 AI images here and a far smaller share of what is actually encountered on
the web, which is dominated by diffusion and transformer models. Recorded as a
real gap rather than adjusted away.

Throughput on WebAssembly is 2.88 images per second, roughly 350 ms per image.
WebGPU was unavailable in headless Chrome; on hardware that exposes an adapter
the backbone runs there and this should improve. Browsing is unaffected either
way, since analysis is lazy, queued two at a time, and cached per URL.

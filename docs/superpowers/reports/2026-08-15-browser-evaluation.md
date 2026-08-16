# Browser evaluation

**Date:** 2026-08-15
**Measured through:** the built extension, driven in a real Chrome by
`eval/run-benchmark.mjs`

## Headline

| Metric | Value |
|---|---|
| **Balanced accuracy at the 0.65 threshold** | **0.8335** |
| Required to qualify | 0.7500 |
| True positive rate (AI detected) | 0.7992 |
| True negative rate (real kept) | 0.8677 |
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

---

# Second pass, after an external review

Three other models reviewed the design. Their suggestions were measured rather
than adopted, and the results split cleanly into what helped, what did not, and
one thing that was simply wrong in the system.

## What was applied

| Change | Effect |
|---|---|
| Average five heads trained from different seeds | +1.0 point |
| Append the log embedding norm to the head input | +0.3 point |
| Weight each generator family equally when fitting the threshold | +0.35 point, BigGAN 0.40 to 0.49 |
| Select the head on unseen generators rather than familiar ones | corrects a selection that measured the wrong thing |
| Raise calibration sharpness | **dead zone 0.33 to 0.097**, balanced accuracy unchanged |
| Reproduce the training downscale before cropping | invisible here, real on the web |

Browser balanced accuracy went from 0.8273 to **0.8335**, and the fraction of
images left in the low-confidence band fell by more than two thirds.

## The most valuable finding was not an accuracy gain

The training cache reduces every image to a longest side of 512 before
anything else. The browser was taking a full-resolution image, often two
thousand pixels wide, straight into the 224 graph in one step. Those are
different resampling cascades and they destroy different frequencies.

The benchmark could not see this, because the benchmark is exported from the
cache and is therefore already capped at 512. Every measurement in this
document was taken on images that had already been through the training
pipeline's downscale. The extension now performs the same reduction before
cropping, so a large web image is treated the way training treated it.

The number here does not move. The behaviour on the images the extension will
actually meet does.

## The sharpness knob was specified and never wired up

The design document called for pushing the score distribution away from the
middle so that the result holds up if the evaluation threshold is read as an
abstention band rather than a decision boundary. That was never implemented,
and a third of all images sat between 0.35 and 0.65.

Because the calibration map holds its crossing at the fitted threshold, raising
the slope cannot move a single prediction. The dead zone fell from 0.33 to
0.097 with balanced accuracy identical to four decimal places.

## What was measured and rejected

**An explicit spectral branch.** The suggestion was that GAN upsampling leaves
periodic high-frequency structure that a 224-pixel semantic embedding cannot
see, and that a small residual-statistics branch would recover it. Nineteen
cheap features were computed over the whole training set.

On its own the branch does what was predicted: BigGAN accuracy 0.733 against
0.42 for the CLIP head. But it flags real images constantly, at 0.309, and
concatenating it to the CLIP embedding drives BigGAN down to **0.053**, worse
than either signal alone. The head learns the spectral signature of the
generators it trained on, and BigGAN's differs, so the feature stops helping
and starts misleading. Late fusion with the weight chosen on held-out data
selected a weight of zero.

Part of the blame is the feature design: its lag-8 autocorrelation and
stride-4 sub-grids sit exactly where JPEG's 8x8 block structure lives, so it
measures compression as much as generation. A better version might work. This
one does not, and it is recorded rather than quietly dropped.

**Replacing CLIP with a discriminatively trained backbone.** Rejected on the
evidence: the cross-generator literature finds that end-to-end fine-tuning
learns generator fingerprints and generalizes worse to unseen generators,
which is the only thing that matters here.

## A hypothesis the data refuted

The review predicted that compression was burying the GAN signal, and that
BigGAN would score worse on augmented images than pristine ones. Measured, it
is the reverse: pristine 0.352, augmented 0.454. Compression is not what is
hiding BigGAN. The geometry of the CLIP embedding is the whole story, which
also explains why a bigger head plateaued.

# Model weight licenses

Every weight file bundled or downloaded by this project is recorded here with
its upstream source and license. The project itself is MIT, so only weights
under a compatible permissive license may be used.

`tests/test_backbone.py::test_every_backbone_declares_a_permissive_license`
enforces this: a backbone whose declared license is outside the permissive set
fails the test suite rather than quietly shipping.

| Artifact | Source | License | Verified on |
|---|---|---|---|
| `backbones/clip-vit-b32/onnx/vision_model.onnx` | [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32), an ONNX export of [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | MIT | 2026-08-14 |
| `backbones/siglip-base-p16/onnx/vision_model.onnx` | [Xenova/siglip-base-patch16-224](https://huggingface.co/Xenova/siglip-base-patch16-224), an ONNX export of [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) | Apache-2.0 | 2026-08-14 |

## Trained artifacts

`models/<backbone>/head.onnx` and `models/<backbone>/calibration.json` are
produced by this repository's training pipeline and are covered by the
project's MIT license.

## Datasets

Training images come from publicly available datasets listed in
`training/datasets/sources.py`. They are used only to fit the classification
head; no dataset image is redistributed by this repository.

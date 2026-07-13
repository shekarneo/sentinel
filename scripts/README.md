# Development Scripts

CLI scripts for module validation, integration testing, and visualization.
All scripts bootstrap the project via `common.py` (adds project root to `sys.path`).

## Shared Helpers (`common.py`)

| Helper | Purpose |
|--------|---------|
| `SCRFD_MODEL_PATH` | Resolved SCRFD ONNX path from configuration |
| `load_image(path)` | Load BGR image with OpenCV |
| `validate_detected_faces(faces)` | Assert SCRFD returned faces |
| `validate_aligned_faces(faces)` | Assert alignment data is populated |

## Integration Tests

| Script | Pipeline |
|--------|----------|
| `test_pipeline.py` | All pipeline profiles |
| `test_verification.py` | SCRFD → Alignment → Embedding → Search → Verification |
| `test_faiss_search.py` | SCRFD → Alignment → Embedding → Search |
| `test_arcface_embedding.py` | SCRFD → Alignment → Embedding |
| `test_assessment.py` | SCRFD → Alignment → Assessment |
| `test_face_alignment.py` | SCRFD → Alignment |
| `test_scrfd_detector.py` | SCRFD detection only |

## Module Tests

| Script | Module |
|--------|--------|
| `test_scrfd_inference.py` | SCRFD ONNX inference |
| `test_preprocess.py` | SCRFD preprocessing |
| `test_bbox_decoder.py` | SCRFD bbox decoder |
| `test_landmark_decoder.py` | SCRFD landmark decoder |
| `test_center_priors.py` | SCRFD center priors |
| `test_similarity_transform.py` | Alignment transform |
| `test_alignment_template.py` | ArcFace landmark template |
| `test_arcface_model.py` | ArcFace model metadata |
| `test_blur.py` | Blur analyzer |
| `test_brightness.py` | Brightness analyzer |
| `test_onnx_engine.py` | ONNX runtime engine |

## Visualizations

| Script | Output |
|--------|--------|
| `visualize_scrfd.py` | Detection overlays |
| `visualize_alignment.py` | Original vs aligned face panels |
| `visualize_assessment.py` | Quality metric panels |
| `visualize_embedding.py` | Embedding stat panels |
| `visualize_search.py` | Top-k search match panels |
| `visualize_verification.py` | Verification decision panels |

## Utilities

| Script | Purpose |
|--------|---------|
| `inspect_scrfd.py` | Print SCRFD ONNX tensor metadata |

## Naming Convention

- `test_<module>.py` — automated validation with printed assertions
- `visualize_<module>.py` — interactive or saved image panels (ESC / N / S keys)

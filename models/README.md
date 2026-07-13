# Models

This directory contains all pretrained AI models used by the Sentinel project.

Models are **not** committed to the repository due to their size and licensing restrictions. Each model should be downloaded from its official source and placed in the appropriate directory.

---

## Directory Structure

```text
models/
│
├── detection/
│   └── scrfd/
│       └── det_10g.onnx
│
├── embedding/
│   └── arcface/
│       └── w600k_r50.onnx
│
├── assessment/
│
├── fraud/
│
└── document/
```

---

## Current Models

### Face Detection

| Model | Purpose | Status |
|--------|---------|--------|
| SCRFD 10G | Face Detection + 5 Facial Landmarks | ✅ |

Expected Path

```text
models/detection/scrfd/det_10g.onnx
```

---

### Face Recognition

| Model | Purpose | Status |
|--------|---------|--------|
| ArcFace (W600K R50) | Face Embedding Extraction | ✅ |

Expected Path

```text
models/embedding/arcface/w600k_r50.onnx
```

---

## Future Models

### Face Assessment

Reserved for future image quality assessment models.

```text
models/assessment/
```

---

### Biometric Fraud Detection

Reserved for:

- Liveness Detection
- Deepfake Detection
- Replay Attack Detection

```text
models/fraud/
```

---

### Document Processing

Reserved for OCR and document understanding models.

```text
models/document/
```

---

## Model Sources

Current models are obtained from the official **InsightFace Model Zoo**.

Only official pretrained models should be used unless otherwise documented.

---

## License

Each pretrained model is distributed under its respective license.

Please review the original model license before using the models in commercial or production environments.

---

## Notes

The Sentinel architecture is inference-backend independent.

Current implementation:

- ONNX Runtime

Future support:

- TensorRT
- OpenVINO
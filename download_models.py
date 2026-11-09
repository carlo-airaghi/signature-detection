#!/usr/bin/env python3
"""Scarica i pesi ONNX del detector di firme da Hugging Face."""

from huggingface_hub import hf_hub_download

MODELS = [
    ("StabRise/signature_detection_v0.1", "model.onnx", "models/StabRise"),
    ("liberty666/yolo11n-chinese-signature", "yolo11n_signature.onnx", "models/liberty666"),
]

if __name__ == "__main__":
    for repo, fname, dest in MODELS:
        path = hf_hub_download(repo, fname, local_dir=dest)
        print("ok:", path)

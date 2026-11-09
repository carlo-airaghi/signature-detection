#!/usr/bin/env python3
"""API web per la signature detection.

Avvio:
    uvicorn app:app --reload --port 8000

POST /api/analyze  (multipart, campo "file": PDF/PNG/JPG)
→ JSON per pagina: campo firma trovato (si/no), firmato (si/no), immagine annotata.
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detect_signatures import DEFAULT_MODEL, YoloOnnxDetector, process_document

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_MB = 25

app = FastAPI(title="Signature Detection")
detector = YoloOnnxDetector(DEFAULT_MODEL)  # caricato una volta sola


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "doc").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Formato non supportato: {suffix or 'sconosciuto'}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File oltre {MAX_UPLOAD_MB} MB")

    with tempfile.TemporaryDirectory() as tmp:
        doc_path = Path(tmp) / f"upload{suffix}"
        doc_path.write_bytes(data)
        try:
            results = process_document(doc_path, DEFAULT_MODEL, Path(tmp) / "out",
                                       detector=detector)
        except Exception as e:  # documento corrotto, password, ecc.
            raise HTTPException(422, f"Impossibile analizzare il documento: {e}")

        pages = []
        for r in results:
            d = asdict(r)
            img_path = d.pop("annotated_image")
            d["annotated_png_base64"] = base64.b64encode(
                Path(img_path).read_bytes()).decode() if img_path else None
            d["field_found"] = len(r.fields) > 0
            d["all_fields_signed"] = bool(r.fields) and all(f.signed for f in r.fields)
            pages.append(d)

    return {
        "filename": file.filename,
        "pages": pages,
        "summary": {
            "fields_total": sum(len(r.fields) for r in results),
            "fields_signed": sum(1 for r in results for f in r.fields if f.signed),
            "fields_unsigned": sum(1 for r in results for f in r.fields if not f.signed),
            "orphan_signatures": sum(len(r.orphan_signatures) for r in results),
            "rejected_marks": sum(len(r.rejected_marks) for r in results),
        },
    }


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

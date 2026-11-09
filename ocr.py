#!/usr/bin/env python3
"""OCR per documenti scansionati/fotografati.

Backend:
  1. Apple Vision (macOS, via pyobjc) - ottimo su foto, supporta l'italiano;
  2. RapidOCR (ONNX, cross-platform) - fallback;
  3. nessuno -> lista vuota (si lavora solo con le firme YOLO).

Output: parole in formato "fitz words": (x0, y0, x1, y1, word, block, line, word_no)
in coordinate pixel dell'immagine.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

MIN_CONFIDENCE = 0.35

_rapidocr = None  # cache del modello


def ocr_words(image: Image.Image) -> list[tuple]:
    """Riconosce il testo e lo restituisce come parole con bounding box (pixel)."""
    if sys.platform == "darwin":
        try:
            return _vision_words(image)
        except ImportError:
            pass
    try:
        return _rapidocr_words(image)
    except ImportError:
        return []


# ---------------------------------------------------------------- Apple Vision

def _vision_words(image: Image.Image) -> list[tuple]:
    import Vision
    from Foundation import NSURL, NSMakeRange

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        url = NSURL.fileURLWithPath_(tmp.name)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(["it-IT", "en-US"])
        req.setUsesLanguageCorrection_(True)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Vision OCR fallito: {err}")
    Path(tmp.name).unlink(missing_ok=True)

    W, H = image.size
    words: list[tuple] = []
    for bi, obs in enumerate(req.results() or []):
        cand = obs.topCandidates_(1)[0]
        if cand.confidence() < MIN_CONFIDENCE:
            continue
        text = cand.string()
        for wi, m in enumerate(re.finditer(r"\S+", text)):
            box = None
            try:
                rect_obs, _ = cand.boundingBoxForRange_error_(
                    NSMakeRange(m.start(), m.end() - m.start()), None)
                if rect_obs is not None:
                    bb = rect_obs.boundingBox()
                    box = (bb.origin.x * W, (1 - bb.origin.y - bb.size.height) * H,
                           (bb.origin.x + bb.size.width) * W, (1 - bb.origin.y) * H)
            except Exception:
                pass
            if box is None:  # fallback: riparto proporzionale sulla riga
                bb = obs.boundingBox()
                lx0, ly0 = bb.origin.x * W, (1 - bb.origin.y - bb.size.height) * H
                lw = bb.size.width * W
                box = (lx0 + lw * m.start() / max(1, len(text)), ly0,
                       lx0 + lw * m.end() / max(1, len(text)), (1 - bb.origin.y) * H)
            words.append((*box, m.group(), bi, 0, wi))
    return words


# ---------------------------------------------------------------- RapidOCR

def _rapidocr_words(image: Image.Image) -> list[tuple]:
    global _rapidocr
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    if _rapidocr is None:
        _rapidocr = RapidOCR()
    res, _ = _rapidocr(np.asarray(image.convert("RGB")))
    words: list[tuple] = []
    for bi, (box, text, score) in enumerate(res or []):
        if float(score) < MIN_CONFIDENCE:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        lx0, ly0, lx1, ly1 = min(xs), min(ys), max(xs), max(ys)
        toks = text.split()
        n_chars = max(1, sum(len(t) for t in toks) + len(toks) - 1)
        cursor = lx0
        for wi, t in enumerate(toks):
            w = (lx1 - lx0) * len(t) / n_chars
            words.append((cursor, ly0, cursor + w, ly1, t, bi, 0, wi))
            cursor += w + (lx1 - lx0) / n_chars
    return words

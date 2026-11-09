#!/usr/bin/env python3
"""Signature detection pipeline.

Dato un documento (PDF o immagine):
  1. trova le firme autografe con un modello YOLO11 pre-addestrato (ONNX, classe "signature");
  2. individua i campi firma (widget AcroForm oppure parole-chiave nel text layer del PDF,
     es. "Firma", "Signature");
  3. incrocia le due cose e dice, per ogni campo, se e' firmato o no.

Output: JSON su stdout/file + immagini annotate per pagina.

Uso:
    python detect_signatures.py documento.pdf -o output/
    python detect_signatures.py scansione.png --model models/StabRise/model.onnx
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ---------------------------------------------------------------- costanti

DEFAULT_MODEL = Path(__file__).parent / "models" / "StabRise" / "model.onnx"
RENDER_DPI = 200  # risoluzione di render per i PDF

# parole chiave che identificano un'etichetta di campo firma (minuscole)
FIELD_KEYWORDS = (
    "firma", "firmato", "firmatario",
    "signature", "signed", "signatory", "unterschrift",
)

# zona candidata attorno all'etichetta, in punti PDF (1 pt = 1/72")
ZONE_LEFT_PAD = 10
ZONE_RIGHT_PAD = 260
ZONE_TOP_PAD = 80
ZONE_BOTTOM_PAD = 25


# ---------------------------------------------------------------- dataclass

SIMPLE_MARK_THRESHOLD = 0.75  # sopra questo score il segno non e' una firma vera


@dataclass
class Detection:
    """Una firma trovata da YOLO (coordinate pixel dell'immagine di pagina)."""
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    simple_mark_score: float | None = None  # ~1.0 = X/spunta/riga, non firma


@dataclass
class SignatureField:
    """Un campo firma individuato nel documento (coordinate pixel)."""
    label: str
    box: tuple[float, float, float, float]
    source: str  # "acroform" | "keyword"
    signed: bool = False
    signature_confidence: float | None = None


@dataclass
class PageResult:
    page: int
    fields: list[SignatureField] = field(default_factory=list)
    orphan_signatures: list[Detection] = field(default_factory=list)
    rejected_marks: list[Detection] = field(default_factory=list)  # X, spunte, righe
    annotated_image: str | None = None


# ---------------------------------------------------------------- YOLO ONNX

class YoloOnnxDetector:
    """Inferenza su export ONNX Ultralytics (output [1, 4+nc, 8400])."""

    def __init__(self, model_path: str | Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta.get("names", "{0: 'signature'}"))
        self.imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = img.shape[:2]
        new_h, new_w = self.imgsz
        r = min(new_h / h, new_w / w)
        uh, uw = round(h * r), round(w * r)
        pad_w, pad_h = (new_w - uw) / 2, (new_h - uh) / 2
        resized = np.array(Image.fromarray(img).resize((uw, uh), Image.BILINEAR))
        out = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
        out[top:top + uh, left:left + uw] = resized
        return out, r, (pad_w, pad_h)

    def __call__(self, image: Image.Image, conf_thres: float = 0.25,
                 iou_thres: float = 0.45) -> list[Detection]:
        img = np.asarray(image.convert("RGB"))
        inp, r, (pad_w, pad_h) = self._letterbox(img)
        blob = inp.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.session.run(None, {self.input_name: blob})[0][0]  # (4+nc, 8400)
        pred = pred.T  # (8400, 4+nc)
        boxes_xywh, scores = pred[:, :4], pred[:, 4:].max(axis=1)
        keep = scores > conf_thres
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]
        if not len(boxes_xywh):
            return []
        # xywh (letterbox) -> xyxy (immagine originale)
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img.shape[1])
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img.shape[0])
        keep_idx = _nms(xyxy, scores, iou_thres)
        return [Detection(tuple(round(float(v), 1) for v in xyxy[i]), round(float(scores[i]), 3))
                for i in keep_idx]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= iou_thres]
    return keep


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


# ---------------------------------------------------------------- filtro anti-X

def simple_mark_score(image: Image.Image, box: tuple[float, float, float, float],
                      max_lines: int = 3, tol: float = 3.0, seed: int = 0) -> float:
    """Frazione di inchiostro del crop spiegabile da pochi segmenti rettilinei.

    Una X, una spunta o una riga ottengono uno score alto (~1.0): sono "simple
    mark", non firme. Una firma corsiva, piena di curve e asole, resta bassa.
    """
    crop = image.convert("L").crop(tuple(int(v) for v in box))
    arr = np.asarray(crop)
    pts = np.argwhere(arr < 128).astype(float)  # (y, x) dei pixel di inchiostro
    if len(pts) < 30:
        return 1.0  # quasi niente inchiostro: sicuramente non e' una firma
    rng = np.random.default_rng(seed)
    if len(pts) > 2000:
        pts = pts[rng.choice(len(pts), 2000, replace=False)]
    total = len(pts)
    remaining = pts
    explained = 0
    for _ in range(max_lines):
        if len(remaining) < max(10, total * 0.05):
            break
        # RANSAC: linea passante per 2 punti con piu' inlier
        best_mask, best_count = None, 0
        for _ in range(250):
            i, j = rng.choice(len(remaining), 2, replace=False)
            p, q = remaining[i], remaining[j]
            d = q - p
            n = np.hypot(*d)
            if n < 5:
                continue
            d /= n
            diff = remaining - p
            dist = np.abs(diff[:, 0] * d[1] - diff[:, 1] * d[0])
            mask = dist < tol
            c = int(mask.sum())
            if c > best_count:
                best_mask, best_count = mask, c
        if best_mask is None or best_count < total * 0.05:
            break
        explained += best_count
        remaining = remaining[~best_mask]
    return explained / total


# ---------------------------------------------------------------- campi firma

def find_fields_pdf_page(page: "fitz.Page", scale: float) -> list[SignatureField]:
    """Campi firma da widget AcroForm + parole chiave nel text layer (coordinate pixel)."""
    fields: list[SignatureField] = []

    # 1) widget AcroForm di tipo firma
    for w in page.widgets() or []:
        if w.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
            r = w.rect * scale
            fields.append(SignatureField(
                label=w.field_name or "signature_widget",
                box=(r.x0, r.y0, r.x1, r.y1), source="acroform"))

    # 2) parole chiave nel testo
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    page_rect = page.rect
    for x0, y0, x1, y1, word, block, line, _wno in words:
        w = word.lower().strip(":.,;()[]")
        if not any(w.startswith(k) for k in FIELD_KEYWORDS):
            continue
        # scarta le keyword dentro un paragrafo: un'etichetta di campo sta su
        # una riga corta (es. "Firma del Cliente"), non in un testo scorrevole
        line_words = [lw for lw in words if lw[5] == block and lw[6] == line]
        if len(line_words) > 5:
            continue
        # etichetta completa = parole sulla stessa riga vicine alla keyword
        label_words = [word]
        for wx0, wy0, wx1, wy1, other, *_ in words:
            if other != word and abs(wy0 - y0) < 3 and 0 < wx0 - x1 < 150:
                label_words.append(other)
        zone = fitz.Rect(
            max(page_rect.x0, x0 - ZONE_LEFT_PAD),
            max(page_rect.y0, y0 - ZONE_TOP_PAD),
            min(page_rect.x1, x1 + ZONE_RIGHT_PAD),
            min(page_rect.y1, y1 + ZONE_BOTTOM_PAD),
        ) * scale
        fields.append(SignatureField(
            label=" ".join(label_words)[:60],
            box=(zone.x0, zone.y0, zone.x1, zone.y1), source="keyword"))

    # dedup: zone keyword quasi sovrapposte
    deduped: list[SignatureField] = []
    for f in fields:
        if not any(_box_iou(f.box, g.box) > 0.6 for g in deduped):
            deduped.append(f)
    return deduped


def _box_iou(a, b) -> float:
    return float(_iou_one_to_many(np.array(a, dtype=float), np.array([b], dtype=float))[0])


def match_fields(fields: list[SignatureField], dets: list[Detection],
                 min_overlap: float = 0.25) -> list[Detection]:
    """Marca i campi come firmati se una firma rilevata vi ricade; ritorna le firme orfane.

    min_overlap: frazione dell'area della firma che deve cadere dentro la zona del campo.
    """
    used = set()
    for f in fields:
        fx1, fy1, fx2, fy2 = f.box
        best, best_frac = None, 0.0
        for i, d in enumerate(dets):
            dx1, dy1, dx2, dy2 = d.box
            ix = max(0.0, min(fx2, dx2) - max(fx1, dx1))
            iy = max(0.0, min(fy2, dy2) - max(fy1, dy1))
            d_area = max(1e-9, (dx2 - dx1) * (dy2 - dy1))
            frac = ix * iy / d_area
            if frac > best_frac:
                best, best_frac = i, frac
        if best is not None and best_frac >= min_overlap:
            f.signed = True
            f.signature_confidence = dets[best].confidence
            used.add(best)
    return [d for i, d in enumerate(dets) if i not in used]


# ---------------------------------------------------------------- rendering

def filter_simple_marks(image: Image.Image,
                        dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
    """Separa le firme vere dai segni semplici (X, spunte, righe)."""
    real, rejected = [], []
    for d in dets:
        d.simple_mark_score = round(simple_mark_score(image, d.box), 3)
        (real if d.simple_mark_score < SIMPLE_MARK_THRESHOLD else rejected).append(d)
    return real, rejected


def annotate(image: Image.Image, fields: list[SignatureField],
             orphans: list[Detection], rejected: list[Detection],
             out_path: Path) -> None:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()

    def box_label(box, color, text):
        draw.rectangle(box, outline=color, width=4)
        tb = draw.textbbox((box[0], box[1] - 28), text, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((box[0], box[1] - 28), text, fill="white", font=font)

    for f in fields:
        if f.signed:
            box_label(f.box, (0, 150, 0), f"FIRMATO {f.signature_confidence:.2f} | {f.label}")
        else:
            box_label(f.box, (200, 0, 0), f"NON FIRMATO | {f.label}")
    for d in orphans:
        box_label(d.box, (0, 90, 200), f"firma fuori campo {d.confidence:.2f}")
    for d in rejected:
        box_label(d.box, (220, 130, 0), f"segno semplice, non firma ({d.simple_mark_score:.2f})")
    img.save(out_path)


# ---------------------------------------------------------------- pipeline

def process_document(doc_path: Path, model_path: Path, out_dir: Path,
                     conf: float = 0.25,
                     detector: YoloOnnxDetector | None = None) -> list[PageResult]:
    if detector is None:
        detector = YoloOnnxDetector(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PageResult] = []

    if doc_path.suffix.lower() == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF non installato: pip install pymupdf")
        doc = fitz.open(doc_path)
        scale = RENDER_DPI / 72
        for pno, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            dets, rejected = filter_simple_marks(image, detector(image, conf_thres=conf))
            fields = find_fields_pdf_page(page, scale)
            orphans = match_fields(fields, dets)
            ann = out_dir / f"{doc_path.stem}_p{pno}.png"
            annotate(image, fields, orphans, rejected, ann)
            results.append(PageResult(pno, fields, orphans, rejected, str(ann)))
        doc.close()
    else:
        image = Image.open(doc_path)
        dets, rejected = filter_simple_marks(image, detector(image, conf_thres=conf))
        # niente text layer: riportiamo solo le firme trovate
        ann = out_dir / f"{doc_path.stem}_annotated.png"
        annotate(image, [], dets, rejected, ann)
        results.append(PageResult(1, [], dets, rejected, str(ann)))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Trova i campi firma e dice se sono firmati.")
    ap.add_argument("document", type=Path, help="PDF o immagine (png/jpg)")
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL,
                    help="pesi ONNX del detector di firme")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"),
                    help="cartella per immagini annotate")
    ap.add_argument("--conf", type=float, default=0.25, help="soglia di confidenza YOLO")
    ap.add_argument("--json", type=Path, help="scrivi il report JSON su file")
    args = ap.parse_args()

    results = process_document(args.document, args.model, args.out, conf=args.conf)
    report = {
        "document": str(args.document),
        "model": str(args.model),
        "pages": [asdict(r) for r in results],
        "summary": {
            "fields_total": sum(len(r.fields) for r in results),
            "fields_signed": sum(1 for r in results for f in r.fields if f.signed),
            "fields_unsigned": sum(1 for r in results for f in r.fields if not f.signed),
            "orphan_signatures": sum(len(r.orphan_signatures) for r in results),
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        args.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

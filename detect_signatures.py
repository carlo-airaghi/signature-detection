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
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ---------------------------------------------------------------- costanti

DEFAULT_MODEL = Path(__file__).parent / "models" / "StabRise" / "model.onnx"
RENDER_DPI = 200  # risoluzione di render per i PDF

# parole chiave che identificano un'etichetta di campo firma (minuscole)
FIELD_KEYWORDS = (
    "firma", "firme", "firmato", "firmatario",
    "signature", "signed", "signatory", "unterschrift",
)

# zona candidata attorno all'etichetta, in punti PDF (1 pt = 1/72"):
# usata solo come fallback quando non c'e' una riga di firma vicina
ZONE_LEFT_PAD = 10
ZONE_RIGHT_PAD = 260
ZONE_TOP_PAD = 80
ZONE_BOTTOM_PAD = 25

# righe di firma: linee orizzontali di larghezza tipica di una firma
SIG_LINE_MIN_W = 70
SIG_LINE_MAX_W = 340
SIG_ZONE_ABOVE = 85   # quanto spazio sopra la riga appartiene alla firma
LABEL_LINE_MAX_DIST = 60  # distanza max etichetta <-> riga di firma


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
    name: str  # nome del campo, es. "CLIENTE" da "Firma del Cliente"
    box: tuple[float, float, float, float]
    source: str  # "acroform" | "keyword" | "line"
    signed: bool = False
    signature_confidence: float | None = None
    weak: bool = False  # etichetta OCR incerta senza riga: vale solo se firmato


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

_NAME_STOPWORDS = {"del", "dello", "della", "dei", "delle", "degli", "di", "da",
                   "il", "lo", "la", "the", "of", "by", "e", "and"}


def _signature_lines(page: "fitz.Page") -> list[tuple[float, float, float]]:
    """Linee orizzontali con larghezza tipica di una riga di firma: (x0, y, x1)."""
    out: list[tuple[float, float, float]] = []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 2:
                    out.append((min(p1.x, p2.x), (p1.y + p2.y) / 2, max(p1.x, p2.x)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 3:
                    out.append((r.x0, (r.y0 + r.y1) / 2, r.x1))
    return [l for l in out if SIG_LINE_MIN_W <= l[2] - l[0] <= SIG_LINE_MAX_W]


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Media locale con finestra k x k (immagine integrale)."""
    k |= 1
    pad = k // 2
    p = np.pad(a, pad, mode="edge").astype(np.float64)
    ii = p.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return s / (k * k)


def _signature_lines_image(image: Image.Image, u: float) -> list[tuple[float, float, float]]:
    """Righe di firma rilevate dai pixel (scansioni/foto senza grafica vettoriale)."""
    g = np.asarray(image.convert("L"), dtype=np.float32)
    H, W = g.shape
    ink = g < _box_mean(g, max(17, int(12 * u))) - 12  # soglia adattiva
    # dilatazione solo orizzontale: salda le interruzioni jpeg senza
    # fondere la linea con la firma che la tocca
    d = ink
    for s in (1, 2, 3):
        d = d | np.roll(ink, s, axis=1) | np.roll(ink, -s, axis=1)
    min_w, max_w = SIG_LINE_MIN_W * u, SIG_LINE_MAX_W * u
    segs: list[tuple[int, float, float]] = []
    for y in range(H):
        xs = np.flatnonzero(d[y])
        if xs.size < min_w * 0.4:
            continue
        cuts = np.flatnonzero(np.diff(xs) > 1)
        start = 0
        for end in [*cuts, xs.size - 1]:
            x0, x1 = float(xs[start]), float(xs[end])
            if x1 - x0 >= min_w * 0.5:  # anche tronconi di linee spezzate
                segs.append((y, x0, x1))
            start = end + 1
    # raggruppa i segmenti su righe adiacenti (linea anche leggermente storta)
    groups: list[list[float]] = []  # [y_min, y_max, x0, x1]
    for y, x0, x1 in segs:
        for grp in groups:
            if y - grp[1] <= 3 and min(grp[3], x1) - max(grp[2], x0) > 0.3 * (x1 - x0):
                grp[1] = y
                grp[2] = min(grp[2], x0)
                grp[3] = max(grp[3], x1)
                break
        else:
            groups.append([y, y, x0, x1])
    # fonde i gruppi collineari separati da piccoli gap: un separatore spezzato
    # dalla foto torna cosi' alla sua larghezza reale (e viene scartato sotto)
    fused = True
    while fused:
        fused = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                ya, yb = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
                gap = max(a[2], b[2]) - min(a[3], b[3])
                if abs(ya - yb) <= 4 and gap <= 30:
                    a[0], a[1] = min(a[0], b[0]), max(a[1], b[1])
                    a[2], a[3] = min(a[2], b[2]), max(a[3], b[3])
                    del groups[j]
                    fused = True
                    break
            if fused:
                break
    max_h = max(6.0, 5.5 * u)
    lines: list[tuple[float, float, float]] = []
    for y_min, y_max, x0, x1 in groups:
        if not (min_w <= x1 - x0 <= max_w) or y_max - y_min > max_h:
            continue
        # una linea disegnata copre quasi tutte le colonne della sua banda;
        # una riga di testo ha i vuoti tra parole e lettere
        band = ink[int(y_min):int(y_max) + 1, int(x0):int(x1) + 1]
        if float(band.any(axis=0).mean()) >= 0.93:
            lines.append((x0, (y_min + y_max) / 2, x1))
    return lines


def _lev(a: str, b: str) -> int:
    """Distanza di Levenshtein (bastano valori piccoli)."""
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _matches_keyword(word: str, fuzzy: bool) -> tuple[bool, bool]:
    """(match esatto, match lasco): il lasco tollera errori OCR ("Finme", "Fiena")."""
    w = word.lower().strip(":.,;()[]'\"")
    if any(w.startswith(k) for k in FIELD_KEYWORDS):
        return True, True
    if fuzzy and len(w) >= 4:
        for k in ("firma", "firme", "firmato", "signature", "signed"):
            if _lev(w, k) <= 2:
                return False, True
    return False, False


def field_name_from_label(label: str, fuzzy: bool = False) -> str:
    """"Firma del Cliente" -> "CLIENTE"; se non resta nulla, l'etichetta intera."""
    words = [w.strip(":.,;()[]_") for w in label.split()]
    kept = [w for w in words
            if w and not any(_matches_keyword(w, fuzzy))
            and w.lower() not in _NAME_STOPWORDS]
    return " ".join(kept).upper() or label.strip(":.,;()[]_ ").upper()


def find_fields_pdf_page(page: "fitz.Page", scale: float) -> list[SignatureField]:
    """Campi firma da widget AcroForm + parole chiave nel text layer (coordinate pixel)."""
    fields: list[SignatureField] = []

    # 1) widget AcroForm di tipo firma
    for w in page.widgets() or []:
        if w.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
            r = w.rect * scale
            fields.append(SignatureField(
                label=w.field_name or "signature_widget",
                name=(w.field_name or "CAMPO FIRMA").upper(),
                box=(r.x0, r.y0, r.x1, r.y1), source="acroform"))

    pr = page.rect
    fields += _fields_from_text(
        page.get_text("words"), _signature_lines(page),
        (pr.x0, pr.y0, pr.x1, pr.y1), scale)
    return _dedupe_sort(fields)


def find_fields_image(image: Image.Image,
                      px_per_pt: float | None = None) -> list[SignatureField]:
    """Campi firma per scansioni/foto: testo via OCR, righe di firma dai pixel."""
    from ocr import ocr_words
    u = px_per_pt or image.width / 595.0  # px per punto PDF (riferimento: A4)
    fields = _fields_from_text(
        ocr_words(image), _signature_lines_image(image, u),
        (0.0, 0.0, float(image.width), float(image.height)),
        scale=1.0, u=u, fuzzy=True)
    return _dedupe_sort(fields)


def _dedupe_sort(fields: list[SignatureField]) -> list[SignatureField]:
    fields.sort(key=lambda f: (f.box[1], f.box[0]))
    deduped: list[SignatureField] = []
    for f in fields:
        if not any(_box_iou(f.box, g.box) > 0.6 for g in deduped):
            deduped.append(f)
    return deduped


def _fields_from_text(words, sig_lines, bounds, scale: float, u: float = 1.0,
                      fuzzy: bool = False) -> list[SignatureField]:
    """Campi firma da parole + righe di firma.

    `u` converte i parametri (espressi in punti PDF) nell'unita' delle
    coordinate in ingresso (1.0 per i pt, px-per-pt per i pixel); `fuzzy`
    attiva la tolleranza agli errori OCR.
    """
    fields: list[SignatureField] = []
    bx0, by0, bx1, by1 = bounds
    line_used = [False] * len(sig_lines)

    def clamp(r: "fitz.Rect") -> "fitz.Rect":
        return fitz.Rect(max(bx0, r.x0), max(by0, r.y0),
                         min(bx1, r.x1), min(by1, r.y1))

    def zone_from_line(label_rect: "fitz.Rect", sl) -> "fitz.Rect":
        lx0, ly, lx1 = sl
        x0, x1 = min(lx0, label_rect.x0) - 5 * u, max(lx1, label_rect.x1) + 5 * u
        if ly >= label_rect.y1 - 2 * u:  # riga sotto l'etichetta: firma tra le due
            y0, y1 = label_rect.y0 - 5 * u, ly + 10 * u
        else:                            # riga sopra l'etichetta: firma sopra la riga
            y0, y1 = ly - SIG_ZONE_ABOVE * u, label_rect.y1 + 5 * u
        return clamp(fitz.Rect(x0, y0, x1, y1))

    def nearest_free_line(label_rect: "fitz.Rect"):
        best, best_d = None, (LABEL_LINE_MAX_DIST + 1) * u
        for i, (lx0, ly, lx1) in enumerate(sig_lines):
            if line_used[i]:
                continue
            overlap = min(lx1, label_rect.x1) - max(lx0, label_rect.x0)
            if overlap < min(40.0 * u, (lx1 - lx0) * 0.3):
                continue
            d = abs(ly - (label_rect.y0 + label_rect.y1) / 2)
            if d < best_d:
                best, best_d = i, d
        return best

    def text_line_info(block: int, line: int):
        lws = sorted((lw for lw in words if lw[5] == block and lw[6] == line),
                     key=lambda t: t[0])
        rect = fitz.Rect(min(t[0] for t in lws), min(t[1] for t in lws),
                         max(t[2] for t in lws), max(t[3] for t in lws))
        return lws, rect, " ".join(t[4] for t in lws)[:60]

    # 1) parole chiave nel testo
    seen_text_lines = set()
    for x0, y0, x1, y1, word, block, line, _wno in words:
        exact, loose = _matches_keyword(word, fuzzy)
        if not loose:
            continue
        if (block, line) in seen_text_lines:
            continue
        seen_text_lines.add((block, line))
        # scarta le keyword dentro un paragrafo: un'etichetta di campo sta su
        # una riga corta (es. "Firma del Cliente"), non in un testo scorrevole
        lws, label_rect, label = text_line_info(block, line)
        if len(lws) > 5:
            continue
        li = nearest_free_line(label_rect)
        weak = False
        if li is not None:
            line_used[li] = True
            zone = zone_from_line(label_rect, sig_lines[li])
        else:
            # zona fissa di ripiego; per i match solo-fuzzy (OCR incerto,
            # tipicamente perche' la firma copre il testo) il campo vale
            # solo se poi risulta firmato
            weak = not exact
            zone = clamp(fitz.Rect(x0 - ZONE_LEFT_PAD * u, y0 - ZONE_TOP_PAD * u,
                                   x1 + ZONE_RIGHT_PAD * u, y1 + ZONE_BOTTOM_PAD * u))
        zone *= scale
        fields.append(SignatureField(
            label=label, name=field_name_from_label(label, fuzzy),
            box=(zone.x0, zone.y0, zone.x1, zone.y1), source="keyword", weak=weak))

    # 2) righe di firma senza keyword: etichettate dal testo corto che le sovrasta
    #    (es. "L'Agente Procuratore" sopra la riga)
    for i, (lx0, ly, lx1) in enumerate(sig_lines):
        if line_used[i]:
            continue
        best_key, best_d = None, 91.0 * u
        for x0, y0, x1, y1, word, block, line, _wno in words:
            if not (ly - 90 * u <= y1 <= ly - 2 * u):
                continue
            if min(lx1, x1) - max(lx0, x0) <= 0:
                continue
            d = ly - y1
            if d < best_d:
                best_key, best_d = (block, line), d
        if best_key is None:
            continue
        lws, label_rect, label = text_line_info(*best_key)
        if len(lws) > 4:  # sopra c'e' un paragrafo, non un'etichetta
            continue
        if any(c.isdigit() for c in label):  # date, importi, CAP: non sono ruoli
            continue
        zone = zone_from_line(label_rect, sig_lines[i]) * scale
        fields.append(SignatureField(
            label=label, name=field_name_from_label(label, fuzzy),
            box=(zone.x0, zone.y0, zone.x1, zone.y1), source="line"))

    return fields


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
            if not fields and not page.get_text("words"):
                # pagina scansionata senza text layer: OCR + righe dai pixel
                fields = find_fields_image(image, px_per_pt=scale)
            orphans = match_fields(fields, dets)
            fields = [f for f in fields if f.signed or not f.weak]
            ann = out_dir / f"{doc_path.stem}_p{pno}.png"
            annotate(image, fields, orphans, rejected, ann)
            results.append(PageResult(pno, fields, orphans, rejected, str(ann)))
        doc.close()
    else:
        image = ImageOps.exif_transpose(Image.open(doc_path)).convert("RGB")
        dets, rejected = filter_simple_marks(image, detector(image, conf_thres=conf))
        fields = find_fields_image(image)  # testo via OCR, righe dai pixel
        orphans = match_fields(fields, dets)
        fields = [f for f in fields if f.signed or not f.weak]
        ann = out_dir / f"{doc_path.stem}_annotated.png"
        annotate(image, fields, orphans, rejected, ann)
        results.append(PageResult(1, fields, orphans, rejected, str(ann)))
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

#!/usr/bin/env python3
"""Genera un PDF di prova: contratto con due campi firma, uno firmato e uno no."""

import math
import random

import fitz

PAGE_W, PAGE_H = fitz.paper_size("a4")


def draw_signature(page: fitz.Page, x: float, y: float, width: float = 170,
                   seed: int = 7) -> None:
    """Disegna uno scarabocchio corsivo che somiglia a una firma autografa."""
    rng = random.Random(seed)
    shape = page.new_shape()
    n_loops = 9
    step = width / n_loops
    pts = []
    cx = x
    for i in range(n_loops + 1):
        amp = rng.uniform(8, 26)
        pts.append(fitz.Point(cx, y + rng.uniform(-amp, amp) * 0.4 - (amp if i % 3 == 0 else 0) * 0.5))
        cx += step * rng.uniform(0.7, 1.3)
    # tratto principale: curve di bezier concatenate
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        c1 = fitz.Point(p0.x + (p1.x - p0.x) * 0.3, p0.y - rng.uniform(10, 30))
        c2 = fitz.Point(p0.x + (p1.x - p0.x) * 0.7, p1.y + rng.uniform(10, 30))
        shape.draw_bezier(p0, c1, c2, p1)
    # svolazzo finale
    last = pts[-1]
    shape.draw_bezier(last, fitz.Point(last.x + 20, last.y - 25),
                      fitz.Point(last.x - 40, last.y - 15),
                      fitz.Point(last.x + 35, last.y + 6))
    # taglio orizzontale sotto, tipico delle firme
    shape.draw_line(fitz.Point(x - 5, y + 12), fitz.Point(x + width * 0.8, y + 8))
    shape.finish(color=(0.05, 0.08, 0.4), width=1.6)
    shape.commit()


def main() -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    text = (
        "CONTRATTO DI FORNITURA SERVIZI\n\n"
        "Tra la societa' Alpha S.r.l., con sede in Milano, Via Roma 1 (di seguito il "
        "Fornitore), e il sig. Mario Rossi, nato a Como il 12/03/1985 (di seguito il "
        "Cliente), si conviene e si stipula quanto segue.\n\n"
        "Art. 1 - Oggetto. Il Fornitore si impegna a erogare i servizi descritti "
        "nell'Allegato A alle condizioni economiche ivi indicate.\n\n"
        "Art. 2 - Durata. Il presente contratto ha durata di 12 mesi a decorrere "
        "dalla data di sottoscrizione, rinnovabile tacitamente.\n\n"
        "Art. 3 - Corrispettivo. Il Cliente corrispondera' al Fornitore l'importo "
        "di Euro 1.200,00 oltre IVA, con pagamento a 30 giorni data fattura.\n\n"
        "Art. 4 - Trattamento dei dati. Le parti si danno reciprocamente atto che i "
        "dati personali saranno trattati ai sensi del Reg. UE 2016/679.\n\n"
        "Letto, approvato e sottoscritto.\n\nMilano, 9 novembre 2026"
    )
    page.insert_textbox(fitz.Rect(60, 60, PAGE_W - 60, 520), text,
                        fontsize=11, fontname="helv")

    # --- campo firma 1: Cliente (FIRMATO) ---
    y1 = 600
    page.draw_line(fitz.Point(70, y1), fitz.Point(280, y1), width=0.8)
    page.insert_text(fitz.Point(70, y1 + 15), "Firma del Cliente", fontsize=10)
    draw_signature(page, 95, y1 - 22, width=150, seed=11)

    # --- campo firma 2: Fornitore (NON firmato) ---
    page.draw_line(fitz.Point(330, y1), fitz.Point(540, y1), width=0.8)
    page.insert_text(fitz.Point(330, y1 + 15), "Firma del Fornitore", fontsize=10)

    doc.save("test_contract.pdf")
    print("scritto test_contract.pdf")


if __name__ == "__main__":
    main()

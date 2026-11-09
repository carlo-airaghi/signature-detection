#!/usr/bin/env python3
"""PDF di prova con tre campi: firma vera, X-mark, vuoto."""

import fitz
from make_test_doc import draw_signature, PAGE_W, PAGE_H


def main() -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_textbox(fitz.Rect(60, 60, PAGE_W - 60, 200),
                        "MODULO DI CONSENSO\n\nIl presente modulo richiede tre firme.",
                        fontsize=11, fontname="helv")

    # campo 1: firma vera
    y = 320
    page.draw_line(fitz.Point(70, y), fitz.Point(280, y), width=0.8)
    page.insert_text(fitz.Point(70, y + 15), "Firma del dichiarante", fontsize=10)
    draw_signature(page, 95, y - 22, width=150, seed=23)

    # campo 2: una semplice X (NON deve contare come firma)
    y = 450
    page.draw_line(fitz.Point(70, y), fitz.Point(280, y), width=0.8)
    page.insert_text(fitz.Point(70, y + 15), "Firma del testimone", fontsize=10)
    sh = page.new_shape()
    sh.draw_line(fitz.Point(140, y - 40), fitz.Point(185, y - 5))
    sh.draw_line(fitz.Point(185, y - 40), fitz.Point(140, y - 5))
    sh.finish(color=(0.05, 0.08, 0.4), width=1.8)
    sh.commit()

    # campo 3: vuoto
    y = 580
    page.draw_line(fitz.Point(70, y), fitz.Point(280, y), width=0.8)
    page.insert_text(fitz.Point(70, y + 15), "Firma del garante", fontsize=10)

    doc.save("test_xmark.pdf")
    print("scritto test_xmark.pdf")


if __name__ == "__main__":
    main()

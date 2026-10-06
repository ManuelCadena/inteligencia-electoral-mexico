#!/usr/bin/env python3
"""Ensamblador del Master Documento (protocolo anti-truncamiento).

Lee docs/masterdoc/assets/plantilla_head.html + sections/p*.html + cierre.html
y produce docs/MASTER_DOCUMENTO.html. Valida balance de etiquetas con html.parser.
"""
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]  # repo root
MD = ROOT / "docs" / "masterdoc"
OUT = ROOT / "docs" / "MASTER_DOCUMENTO.html"

class Checker(HTMLParser):
    VOID = {"meta", "link", "br", "hr", "img", "input", "col", "path", "circle",
            "rect", "line", "polyline", "polygon", "ellipse", "stop", "use", "area"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors = [], []
    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()))
    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> sin apertura @ {self.getpos()}")
            return
        open_tag, pos = self.stack.pop()
        if open_tag != tag:
            self.errors.append(f"esperado </{open_tag}> (abierto @ {pos}), vino </{tag}> @ {self.getpos()}")

def main():
    head = (MD / "assets" / "plantilla_head.html").read_text(encoding="utf-8")
    css = (MD / "assets" / "harvard.css").read_text(encoding="utf-8")
    parts = sorted((MD / "sections").glob("p*.html"))
    tail = (MD / "sections" / "cierre.html").read_text(encoding="utf-8")
    if not parts:
        sys.exit("No hay secciones p*.html")
    html = head.replace("/*__CSS__*/", css)
    html += "\n".join(p.read_text(encoding="utf-8") for p in parts)
    html += tail
    c = Checker()
    c.feed(html)
    if c.stack:
        c.errors.append(f"sin cerrar: {[t for t,_ in c.stack]}")
    OUT.write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"[build] secciones: {[p.name for p in parts]}")
    print(f"[build] {OUT.name}: {kb:.1f} KB")
    if c.errors:
        print("[VALIDACIÓN-FALLÓ]")
        for e in c.errors[:30]:
            print(" -", e)
        sys.exit(2)
    print("[VALIDACIÓN-OK] balance de etiquetas correcto")

if __name__ == "__main__":
    main()

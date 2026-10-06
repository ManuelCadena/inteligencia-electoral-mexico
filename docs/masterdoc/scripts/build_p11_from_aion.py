#!/usr/bin/env python3
"""Extract AION-generated pedagogical theory snippets from overflow JSON files,
clean double-encoded UTF-8 mojibake, and produce p11_teoria_pedagogica.html."""
from pathlib import Path
import re, html, sys

ROOT = Path(__file__).resolve().parents[3]
OUT_MD = ROOT / "docs" / "masterdoc" / "sections" / "p11_teoria_pedagogica.md"
OUT_HTML = ROOT / "docs" / "masterdoc" / "sections" / "p11_teoria_pedagogica.html"

FILES = [
    ("DL 1-7", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/1e5c230f/content.txt"),
    ("DL 8-14", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/de3d6e65/content.txt"),
    ("NLP 15-20", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/0a469c97/content.txt"),
    ("NLP 21-26", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/ca98a7e9/content.txt"),
]

def fix_text(txt: str) -> str:
    try:
        return txt.encode('latin-1').decode('utf-8')
    except UnicodeError:
        return txt

def extract_output(raw: str) -> str:
    m = re.search(r'"output":\s*"(.*?)"\s*,\s*"pipeline"', raw, re.DOTALL)
    if not m:
        return ""
    out = m.group(1)
    out = out.encode('utf-8').decode('unicode_escape')
    return fix_text(out)

def latex_to_html(eq: str) -> str:
    # Keep formulas as readable plain code: escape HTML, remove display delimiters,
    # and convert a few common LaTeX commands to plain symbols.
    s = html.escape(eq.strip())
    s = re.sub(r'^\\\[|\\\]$', '', s).strip()
    s = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)
    s = s.replace('\\frac', '').replace('\\sqrt', '√')
    s = s.replace('\\cdot', '·').replace('\\times', '×').replace('\\odot', '⊙')
    s = s.replace('\\sigma', 'σ').replace('\\mu', 'μ').replace('\\epsilon', 'ε')
    s = s.replace('\\sum', 'Σ').replace('\\exp', 'exp').replace('\\log', 'log')
    s = s.replace('\\max', 'max').replace('\\min', 'min').replace('\\mid', '|')
    return s

def md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'\\\((.+?)\\\)', lambda m: f'<code>{latex_to_html(m.group(1))}</code>', text)
    return text

def md_to_html(md: str) -> str:
    # Drop HTML comment lines before processing
    lines = [ln for ln in md.splitlines() if not ln.strip().startswith('<!--')]
    blocks = []
    i = 0
    cur_para = []
    topic_counter = 0
    def flush_para():
        if not cur_para:
            return
        text = " ".join(cur_para)
        text = md_inline(text)
        blocks.append(f"<p>{text}</p>")
        cur_para.clear()
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("## "):
            flush_para()
            if topic_counter > 0:
                blocks.append("</div>\n</div>")
            topic_counter += 1
            title = md_inline(line[3:].strip())
            blocks.append(f"<div class=\"lecture\" id=\"tp{topic_counter}\">\n<header>\n<div class=\"wk\"><span class=\"w\">Topic</span></div>\n<h3>{title}</h3>\n</header>\n<div class=\"body\">")
        elif line.strip() == "":
            flush_para()
        elif line.strip().startswith("\\["):
            flush_para()
            eq_lines = [line.strip()]
            while i < len(lines) and "\\]" not in lines[i]:
                i += 1
                eq_lines.append(lines[i].rstrip())
            eq = " ".join(eq_lines)
            eq = re.sub(r'^\\\[|\\\]$', '', eq).strip()
            eq_html = latex_to_html(eq)
            blocks.append(f'<div class="formula">{eq_html}</div>')
        else:
            cur_para.append(line)
        i += 1
    flush_para()
    blocks.append("</div>\n</div>")
    return "\n".join(blocks)

def main():
    full_md = []
    for label, path in FILES:
        raw = Path(path).read_text(encoding='utf-8')
        out = extract_output(raw)
        if not out:
            print(f"WARN: no output in {label}")
            continue
        full_md.append(f"<!-- ======== {label} ======== -->\n")
        full_md.append(out)
        full_md.append("\n")
    OUT_MD.write_text("\n".join(full_md), encoding='utf-8')

    body_html = md_to_html("\n".join(full_md))
    html_doc = f'''<!-- ============ PART VIII — PEDAGOGICAL THEORY DEEP-DIVE ============ -->
<div class="part" id="p8">
<div class="wrap">
<div class="parthead">
<div class="pn">Part VIII</div>
<h2>Pedagogical deep-dive: theoretical frameworks with everyday analogies</h2>
<p>This part expands every theoretical framework from Parts III and IV into a
self-contained explanation. Each segment assumes no prior knowledge, expands every acronym at
first use, explains the core mechanism from first principles, and provides an everyday analogy.
It is meant to be read <em>before</em> the corresponding notebook, so that the reader understands
<em>why</em> the model exists and <em>how</em> it maps onto the Mexican electoral dataset.
<span class="badge b-syl">SYLLABUS-CONFIRMED</span></p>
</div>

{body_html}
</div>
</div>
'''
    OUT_HTML.write_text(html_doc, encoding='utf-8')
    print(f"[p11] wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")
    print(f"[p11] wrote {OUT_HTML} ({OUT_HTML.stat().st_size} bytes)")

if __name__ == "__main__":
    main()

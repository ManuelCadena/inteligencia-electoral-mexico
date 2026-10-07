#!/usr/bin/env python3
"""Clean stray lecture IDs and convert raw LaTeX-like math inside <code> tags
in p04-p07 to HTML sub/sup notation."""
from pathlib import Path
import re, sys

ROOT = Path(__file__).resolve().parents[3]
SECTIONS = ROOT / "docs" / "masterdoc" / "sections"
FILES = ["p04_dl_a.html", "p05_dl_b.html", "p06_nlp_a.html", "p07_nlp_b.html"]

# Strong markers that suggest a <code> block is math, not a file/identifier.
MATH_MARKERS = re.compile(
    r'[σθη∇∈Σλ√←·⊙‖²μενφΦαβγδ\^]|'
    r'≤|≥|≠|≈|argmin|argmax|softmax|exp\(|log\(|max\(|min\(',
    re.IGNORECASE,
)

GREEK = 'ηθφσδμενλγβαΣΓŷ'

def _convert_one_pass(text: str) -> str:
    # Superscripts with braces, brackets, or single char after a variable (not mid-word)
    text = re.sub(r'\^\{([^{}]+)\}', r'<sup>\1</sup>', text)
    text = re.sub(r'\^\[([^\]]+)\]', r'<sup>[\1]</sup>', text)
    # Single-letter or multi-digit superscripts (R^d, R^768) not preceded by another alnum.
    text = re.sub(r'(?<![A-Za-z0-9])([A-Za-z' + GREEK + '])\\^([0-9]+|[A-Za-z])', r'\1<sup>\2</sup>', text)
    # Subscripts with braces
    text = re.sub(r'_\{([^{}]+)\}', r'<sub>\1</sub>', text)
    # Single-char subscripts after a variable not preceded by another letter.
    # This avoids converting mid-identifier underscores in file names (e.g. transfer_federal).
    text = re.sub(r'(?<![A-Za-z])([A-Za-z' + GREEK + r'])_([A-Za-z0-9])', r'\1<sub>\2</sub>', text)
    # Partial derivatives like ∂L/∂W^[l] after superscript conversion
    text = re.sub(r'∂([a-zA-Z])<sup>(\[[^\]]+\])</sup>', r'∂\1<sup>\2</sup>', text)
    text = text.replace('²', '<sup>2</sup>')
    return text

def math_to_html(text: str) -> str:
    # Repeat to handle nested subscripts, e.g. A_{y_{t-1},y_t}
    prev = None
    while prev != text:
        prev = text
        text = _convert_one_pass(text)
    return text

def convert_global_math(text: str) -> str:
    """Convert sub/superscript patterns in text outside HTML tags and <code> blocks."""
    parts = re.split(r'(<[^>]+>)', text)
    in_code = False
    out = []
    for part in parts:
        if part.startswith('<code') and part.endswith('>'):
            in_code = True
        elif part == '</code>':
            in_code = False
            out.append(part)
            continue
        if not in_code and not part.startswith('<'):
            part = math_to_html(part)
        out.append(part)
    return ''.join(out)

def clean_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    original = text

    # 1. Remove stray IDs injected after lecture divs: e.g. <div class="lecture" id="dl1">dl1
    text = re.sub(r'(<div class="lecture" id="(dl\d+|nlp\d+)">)\2', r'\1', text)

    # 2. Convert math-like <code> contents
    def convert_code(m: re.Match) -> str:
        content = m.group(1)
        if not MATH_MARKERS.search(content):
            return m.group(0)
        return f'<code>{math_to_html(content)}</code>'

    text = re.sub(r'<code>([^<]+)</code>', convert_code, text)

    # 3. Convert math in prose / formula divs, protecting already-converted <code> blocks
    text = convert_global_math(text)

    path.write_text(text, encoding="utf-8")
    changed = original != text
    return 1 if changed else 0

def main():
    total = 0
    for fname in FILES:
        total += clean_file(SECTIONS / fname)
        print(f"[ok] {fname}")
    print(f"[summary] {total} files modified")

if __name__ == "__main__":
    main()

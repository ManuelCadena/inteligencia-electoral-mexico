#!/usr/bin/env python3
"""Extract AION-generated pedagogical theory snippets from overflow JSON files,
clean double-encoded UTF-8 mojibake, and produce p11_teoria_pedagogica.html."""
from pathlib import Path
import re, html, sys

ROOT = Path(__file__).resolve().parents[3]
OUT_MD = ROOT / "docs" / "masterdoc" / "sections" / "p11_teoria_pedagogica.md"
OUT_HTML = ROOT / "docs" / "masterdoc" / "sections" / "p11_teoria_pedagogica.html"

FILES = [
    ("DL 1-5", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/eb59505f/content.txt"),
    ("DL 6-10", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/772ef778/content.txt"),
    ("DL/NLP 11-15", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/9e17c04d/content.txt"),
    ("NLP 16-20", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/d2a99a10/content.txt"),
    ("NLP 21-26", "/var/folders/kn/x8t7l4g553l9_j0bnxl704nw0000gn/T/devin-overflows-501/4dd1c652/content.txt"),
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
    # bold / italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # inline math
    text = re.sub(r'\\\((.+?)\\\)', lambda m: f'<code>{latex_to_html(m.group(1))}</code>', text)
    return text

def render_list_items(items, ordered=False):
    tag = 'ol' if ordered else 'ul'
    out = [f'<{tag}>']
    for item in items:
        out.append(f'<li>{md_inline(item)}</li>')
    out.append(f'</{tag}>')
    return '\n'.join(out)

def render_table(rows):
    # rows: list of lists of cell strings; first row header
    out = ['<table class="compact">']
    out.append('<tr>' + ''.join(f'<th>{md_inline(c)}</th>' for c in rows[0]) + '</tr>')
    for row in rows[2:]:
        out.append('<tr>' + ''.join(f'<td>{md_inline(c)}</td>' for c in row) + '</tr>')
    out.append('</table>')
    return '\n'.join(out)

def md_to_html(md: str) -> str:
    lines = [ln for ln in md.splitlines() if not ln.strip().startswith('<!--')]
    blocks = []
    i = 0
    topic_counter = 0

    def flush_para(buf):
        if not buf:
            return
        text = ' '.join(buf)
        blocks.append(f"<p>{md_inline(text)}</p>")
        buf.clear()

    para_buf = []
    # Standard subsection labels used by the AION prompts
    SUBSECTIONS = {
        'why we care', 'one-sentence intuition', 'how it works',
        'everyday analogy', 'on our mexican elections', 'key paper'
    }
    while i < len(lines):
        line = lines[i].rstrip()
        # horizontal rule
        if re.fullmatch(r'\s*[-*]{3,}\s*', line):
            flush_para(para_buf)
            i += 1
            continue
        topic_match = re.match(r'^(#+)\s*(\d+)\.\s+(.*)$', line)
        if topic_match:
            flush_para(para_buf)
            if topic_counter > 0:
                blocks.append('</div>\n</div>')
            topic_counter += 1
            title = md_inline(topic_match.group(3).strip())
            blocks.append(f'<div class="lecture" id="tp{topic_counter}">\n<header>\n<div class="wk"><span class="w">Topic</span></div>\n<h3>{title}</h3>\n</header>\n<div class="body">')
            i += 1
            continue
        heading_match = re.match(r'^(#{2,4})\s+(.*)$', line)
        if heading_match:
            flush_para(para_buf)
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            tag = {2: 'h4', 3: 'h5', 4: 'h6'}.get(level, 'h4')
            blocks.append(f"<{tag}>{md_inline(text)}</{tag}>")
            i += 1
            continue
        elif line.strip() == '':
            flush_para(para_buf)
        elif line.strip().startswith('\\['):
            flush_para(para_buf)
            eq_lines = [line.strip()]
            while i < len(lines) and '\\]' not in lines[i]:
                i += 1
                eq_lines.append(lines[i].rstrip())
            eq = ' '.join(eq_lines)
            eq = re.sub(r'^\\\[|\\\]$', '', eq).strip()
            eq_html = latex_to_html(eq)
            blocks.append(f'<div class="formula">{eq_html}</div>')
        elif re.match(r'^\s*[-*]\s+', line):
            flush_para(para_buf)
            items = []
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(re.sub(r'^\s*[-*]\s+', '', lines[i]))
                i += 1
            blocks.append(render_list_items(items, ordered=False))
            continue  # already advanced
        elif re.match(r'^\s*\d+\.\s+', line):
            flush_para(para_buf)
            items = []
            while i < len(lines) and re.match(r'^\s*\d+\.\s+', lines[i]):
                items.append(re.sub(r'^\s*\d+\.\s+', '', lines[i]))
                i += 1
            blocks.append(render_list_items(items, ordered=True))
            continue
        elif line.strip().startswith('|'):
            flush_para(para_buf)
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            # parse cells, drop separator row if present
            rows = [re.split(r'\s*\|\s*', ln.strip('|').strip()) for ln in table_lines]
            rows = [r for r in rows if any(c.strip() for c in r)]
            if rows and len(rows) >= 1:
                blocks.append(render_table(rows))
            continue
        else:
            para_buf.append(line)
        i += 1
    flush_para(para_buf)
    if topic_counter > 0:
        blocks.append('</div>\n</div>')
    return '\n'.join(blocks)

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
        full_md.append('\n')
    OUT_MD.write_text(''.join(full_md), encoding='utf-8')

    body_html = md_to_html(''.join(full_md))

    html_doc = f'''<!-- ============ PART VIII — PEDAGOGICAL THEORY DEEP-DIVE ============ -->
<div class="part" id="p8">
<div class="wrap">
<div class="parthead">
<div class="pn">Part VIII</div>
<h2>Pedagogical deep-dive: theoretical frameworks explained like a Harvard lecture</h2>
<p>Each segment below is written as if explaining the idea to a high-school student with no
prior exposure. For every model we answer three questions in plain English:
<strong>Why do we care?</strong> <strong>How does it actually work?</strong> and
<strong>How does it apply to Mexican elections?</strong> Acronyms are expanded at first use,
and every explanation is paired with a concrete everyday analogy. Read this part before any
notebook to understand the intuition behind the code.</p>
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

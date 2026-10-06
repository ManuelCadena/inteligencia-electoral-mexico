#!/usr/bin/env python3
"""Insert a pedagogical pointer to Part VIII after each lecture's
<h4>Theoretical framework</h4> heading in p04-p07 sections."""
from pathlib import Path
import re, sys

ROOT = Path(__file__).resolve().parents[3]
SECTIONS = ROOT / "docs" / "masterdoc" / "sections"

MAP = {
    "p04_dl_a.html": {
        "dl1": ([1], "MLP"),
        "dl2": ([2], "PyTorch and automatic differentiation"),
        "dl3": ([3], "regularization"),
        "dl4": ([4], "CNNs"),
        "dl5": ([5], "transfer learning"),
        "dl6": ([6], "autoencoders and VAEs"),
    },
    "p05_dl_b.html": {
        "dl7": ([7], "Transformers"),
        "dl8": ([8], "automatic speech recognition"),
        "dl9": ([9], "LSTM/GRU and seq2seq"),
        "dl10": ([10, 11, 12], "LLMs, RAG and agentic workflows"),
        "dl11": ([13, 14], "generative and state-space models"),
    },
    "p06_nlp_a.html": {
        "nlp1": ([15, 16], "BoW/TF-IDF and text RNNs"),
        "nlp3": ([17, 18], "BPE tokenization and n-grams"),
        "nlp5": ([19, 20], "word embeddings and t-SNE/UMAP"),
    },
    "p07_nlp_b.html": {
        "nlp7": ([21], "LDA topic modeling"),
        "nlp8": ([22], "structural topic models"),
        "nlp9": ([23, 24], "text classification/stance and NER"),
        "nlp11": ([25, 26], "CRFs and BETO/BERT"),
    },
}

def link_text(topic_ids, label):
    links = ", ".join(f'<a href="#tp{n}">Topic {n}</a>' for n in topic_ids)
    if len(topic_ids) > 1:
        seg = "those short segments"
    else:
        seg = "that short segment"
    return (
        f'<p class="small pedagogy-pointer"><strong>Need a gentler start?</strong> '
        f'Part VIII explains the concepts behind this section from first principles '
        f'(why they matter, how they work, and an everyday analogy) in {links} — {label}. '
        f'Read {seg} first if any term below feels unfamiliar.</p>'
    )

def process_file(fname, mapping):
    path = SECTIONS / fname
    text = path.read_text(encoding="utf-8")
    # Split by lecture divs while preserving them.
    pattern = re.compile(r'(<div class="lecture" id="(dl\d+|nlp\d+)">)')
    parts = pattern.split(text)
    # parts: [pre, '<div...>', id, body, '<div...>', id, body, ...]
    out = [parts[0]]
    for i in range(1, len(parts), 3):
        div_open = parts[i]
        lid = parts[i+1]
        body = parts[i+2] if i+2 < len(parts) else ""
        topic_ids, label = mapping.get(lid, (None, None))
        if topic_ids:
            note = link_text(topic_ids, label)
            # Insert after the first <h4>Theoretical framework</h4> in the body
            body_new = re.sub(
                r'(<h4>Theoretical framework</h4>)(?!\n<p class="small pedagogy-pointer")',
                r'\1\n' + note,
                body,
                count=1,
            )
            if body_new is body:
                print(f"WARN: {fname} {lid} has no <h4>Theoretical framework</h4>")
            body = body_new
        out.extend([div_open, lid, body])
    path.write_text("".join(out), encoding="utf-8")
    print(f"[ok] {fname}")

def main():
    for fname, mapping in MAP.items():
        backup = SECTIONS / (fname + ".bak." + Path(__file__).stat().st_mtime.__str__())
        # no actual backup here; rely on git + caller backup if desired
        process_file(fname, mapping)

if __name__ == "__main__":
    main()

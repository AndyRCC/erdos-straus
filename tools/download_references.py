#!/usr/bin/env python3
"""Downloads the PDFs listed in references/README.md into references/ and extracts their text into references/text/."""
import os
import sys
import urllib.request

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
PDFS = {
    "arXiv_1107.1010.pdf": "https://arxiv.org/pdf/1107.1010",
    "arXiv_1406.6307.pdf": "https://arxiv.org/pdf/1406.6307",
    "arXiv_2509.00128.pdf": "https://arxiv.org/pdf/2509.00128",
    "Pomerance_esconj_slides.pdf": "https://math.dartmouth.edu/~carlp/esconj.pdf",
    "arXiv_2608.24035.pdf": "https://arxiv.org/pdf/2608.24035",
    "arXiv_2605.23601.pdf": "https://arxiv.org/pdf/2605.23601",
    "arXiv_2609.29250.pdf": "https://arxiv.org/pdf/2609.29250",
    "arXiv_2610.32302.pdf": "https://arxiv.org/pdf/2610.32302",
    "arXiv_2502.20935.pdf": "https://arxiv.org/pdf/2502.20935",
    "arXiv_2602.20036.pdf": "https://arxiv.org/pdf/2602.20036",
    "arXiv_1908.02526.pdf": "https://arxiv.org/pdf/1908.02526",
    "arXiv_math_0210333.pdf": "https://arxiv.org/pdf/math/0210333",
    "arXiv_2504.20289.pdf": "https://arxiv.org/pdf/2504.20289",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.join(HERE, "text"), exist_ok=True)
    for name, url in PDFS.items():
        dest_path = os.path.join(HERE, name)
        if not os.path.exists(dest_path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-client)"})
            with urllib.request.urlopen(req) as r, open(dest_path, "wb") as f:
                f.write(r.read())
            print("downloaded:", name)
        try:
            from pypdf import PdfReader
        except ImportError:
            print("  (no pypdf: text not extracted)")
            continue
        page_texts = [(p.extract_text() or "") for p in PdfReader(dest_path).pages]
        with open(os.path.join(HERE, "text", name[:-4] + ".txt"), "w", encoding="utf-8") as out:
            for i, p in enumerate(page_texts, 1):
                out.write(f"\n\n===== page {i} =====\n{p}")


if __name__ == "__main__":
    main()

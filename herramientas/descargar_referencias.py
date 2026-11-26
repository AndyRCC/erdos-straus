#!/usr/bin/env python3
"""Descarga los PDFs de referencias/README.md a referencias/ y extrae su texto a referencias/texto/."""
import os
import sys
import urllib.request

AQUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "referencias")
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
    os.makedirs(os.path.join(AQUI, "texto"), exist_ok=True)
    for nombre, url in PDFS.items():
        destino = os.path.join(AQUI, nombre)
        if not os.path.exists(destino):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-client)"})
            with urllib.request.urlopen(req) as r, open(destino, "wb") as f:
                f.write(r.read())
            print("descargado:", nombre)
        try:
            from pypdf import PdfReader
        except ImportError:
            print("  (sin pypdf: no se extrae el texto)")
            continue
        paginas = [(p.extract_text() or "") for p in PdfReader(destino).pages]
        with open(os.path.join(AQUI, "texto", nombre[:-4] + ".txt"), "w", encoding="utf-8") as out:
            for i, p in enumerate(paginas, 1):
                out.write(f"\n\n===== página {i} =====\n{p}")


if __name__ == "__main__":
    main()

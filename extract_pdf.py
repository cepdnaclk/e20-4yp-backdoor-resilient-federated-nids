import subprocess, sys

# Try pypdf first (no install needed if already present), else fall back to pdfplumber, else pymupdf
def try_pypdf():
    from pypdf import PdfReader
    reader = PdfReader(r"docs\resources\E2008.pdf")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print(f"\n=== SLIDE {i+1} ===\n{text}")

def try_pdfplumber():
    import pdfplumber
    with pdfplumber.open(r"docs\resources\E2008.pdf") as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                print(f"\n=== SLIDE {i+1} ===\n{text}")

for fn in [try_pypdf, try_pdfplumber]:
    try:
        fn()
        sys.exit(0)
    except ImportError:
        continue
    except Exception as e:
        print(f"Error with {fn.__name__}: {e}")
        continue

print("No PDF library available. Trying pip install pypdf...")

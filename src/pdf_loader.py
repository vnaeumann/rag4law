#src.pdf_loader.py

import fitz  # PyMuPDF
import re

import re

def clean_text(text):
    text = text.replace("Transcribed by TERES", "")

    # fix spaced-out uppercase names
    text = re.sub(r"\b([A-Z])\s+(?=[A-Z]\b)", r"\1", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    
    doc_name = getattr(file, "name", "uploaded.pdf")


    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        text = clean_text(text)


        if text.strip():
            pages.append({
                "doc_name":doc_name,
                "page": i,
                "text": text
            })
        

    return pages




import io
import requests
import pymupdf


def download_and_extract_pdf(pdf_url: str) -> str:
    """
    Download a PDF from arXiv and extract its text.
    """

    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    pdf_document = pymupdf.open(
        stream=io.BytesIO(response.content),
        filetype="pdf",
    )

    pages_text = []

    for page in pdf_document:
        pages_text.append(page.get_text())

    pdf_document.close()

    text = "\n".join(pages_text).strip()

    if not text:
        raise ValueError("No text could be extracted from the PDF.")

    return text
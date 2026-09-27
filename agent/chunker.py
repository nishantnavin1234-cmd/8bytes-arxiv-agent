def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """
    Split paper text into overlapping chunks.

    Reference sections are excluded so that bibliography entries
    do not dominate semantic retrieval for paper QA.
    """

    if not text.strip():
        return []

    # Remove the references/bibliography section.
    reference_markers = [
        "\nReferences\n",
        "\nREFERENCES\n",
        "\nBibliography\n",
        "\nBIBLIOGRAPHY\n",
    ]

    for marker in reference_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                }
            )
            chunk_id += 1

        start += chunk_size - overlap

    return chunks
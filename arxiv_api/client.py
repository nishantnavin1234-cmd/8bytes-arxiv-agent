import arxiv


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv and return basic metadata for matching papers.
    """

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    try:
        results = client.results(search)

        papers = []

        for result in results:
            papers.append(
                {
                    "id": result.get_short_id(),
                    "title": result.title.strip(),
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary.strip(),
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "arxiv_url": result.entry_id,
                }
            )

        return papers

    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve papers from arXiv: {exc}"
        ) from exc

def get_paper_by_id(paper_id: str) -> dict:
    """
    Retrieve a specific arXiv paper by its ID.
    """

    client = arxiv.Client(
        page_size=1,
        delay_seconds=3,
        num_retries=3,
    )

    search = arxiv.Search(
        id_list=[paper_id],
    )

    try:
        results = list(client.results(search))

        if not results:
            raise ValueError(
                f"No arXiv paper found for ID '{paper_id}'."
            )

        result = results[0]

        return {
            "id": result.get_short_id(),
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "abstract": result.summary.strip(),
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "categories": result.categories,
            "arxiv_url": result.entry_id,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve arXiv paper '{paper_id}': {exc}"
        ) from exc
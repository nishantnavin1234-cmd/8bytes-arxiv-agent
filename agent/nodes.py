import re

from .state import AgentState
from arxiv_api.client import search_arxiv, get_paper_by_id
from arxiv_api.pdf_parser import download_and_extract_pdf
from agent.chunker import chunk_text
from agent.vector_store import VectorStore


def query_understanding_node(state: AgentState) -> AgentState:
    """
    Determine whether the user provided a specific arXiv paper
    or a natural-language research topic.
    """

    query = state.input_query.strip()

    # arXiv paper ID formats:
    # New format: 2401.12345 or 2401.12345v2
    # Old format: hep-th/9901001 or hep-th/9901001v2
    paper_id_pattern = (
        r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|"
        r"[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$"
    )

    # arXiv URLs
    arxiv_url_pattern = r"^https?://arxiv\.org/(?:abs|pdf)/.+"

    if re.match(paper_id_pattern, query) or re.match(
        arxiv_url_pattern, query
    ):
        state.query_type = "paper"
    else:
        state.query_type = "topic"

    state.status = "query_understood"

    return state


def retrieve_papers_node(state: AgentState) -> AgentState:
    """
    Retrieve papers from arXiv.

    For a specific paper query, retrieve the exact paper by ID.
    For a topic query, search arXiv for candidate papers.
    """

    try:
        if state.query_type == "paper":
            query = state.input_query.strip().rstrip("/")

            if query.startswith("http://") or query.startswith("https://"):
                query = query.split("/")[-1]

                if query.endswith(".pdf"):
                    query = query[:-4]

            paper = get_paper_by_id(query)

            state.candidate_papers = [paper]

        else:
            papers = search_arxiv(
                state.input_query,
                max_results=5,
            )

            state.candidate_papers = papers

        state.status = "papers_retrieved"

    except Exception as exc:
        state.errors.append(str(exc))
        state.status = "retrieval_failed"

    return state

def select_paper_node(state: AgentState) -> AgentState:
    """
    Select the most relevant paper from the retrieved candidates.
    For topic searches, rank papers using title and abstract overlap.
    For specific paper queries, select the matching paper directly.
    """

    if not state.candidate_papers:
        state.status = "selection_failed"
        state.errors.append("No candidate papers available for selection.")
        return state

    query = state.input_query.strip().lower()

    if state.query_type == "paper":
        query = query.rstrip("/").split("/")[-1].removesuffix(".pdf")

        for paper in state.candidate_papers:
            if paper.get("id", "").lower() == query.lower():
                state.selected_paper = paper
                state.status = "paper_selected"
                return state

        state.status = "selection_failed"
        state.errors.append(
            f"Requested paper '{query}' was not found in the retrieved results."
        )
        return state

    query_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query)
        if len(term) > 2
    }

    best_paper = None
    best_score = -1

    for paper in state.candidate_papers:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()

        title_terms = set(re.findall(r"[a-zA-Z0-9]+", title))
        abstract_terms = set(re.findall(r"[a-zA-Z0-9]+", abstract))

        title_matches = len(query_terms & title_terms)
        abstract_matches = len(query_terms & abstract_terms)

        score = (title_matches * 3) + abstract_matches

        if score > best_score:
            best_score = score
            best_paper = paper

    if best_paper is None:
        state.status = "selection_failed"
        state.errors.append("Unable to select a relevant paper.")
        return state

    state.selected_paper = best_paper
    state.status = "paper_selected"

    return state

def fetch_and_parse_node(state: AgentState) -> AgentState:
    """
    Download the selected paper's PDF and extract its text.
    """

    if not state.selected_paper:
        state.status = "parse_failed"
        state.errors.append(
            "No paper has been selected for PDF parsing."
        )
        return state

    pdf_url = state.selected_paper.get("pdf_url")

    if not pdf_url:
        state.status = "parse_failed"
        state.errors.append(
            "Selected paper does not contain a PDF URL."
        )
        return state

    try:
        state.paper_text = download_and_extract_pdf(pdf_url)
        state.status = "paper_parsed"

    except Exception as exc:
        state.errors.append(
            f"Failed to fetch or parse PDF: {exc}"
        )
        state.status = "parse_failed"

    return state

def chunk_paper_node(state: AgentState) -> AgentState:
    """
    Split the parsed paper text into overlapping chunks.
    """

    if not state.paper_text.strip():
        state.status = "chunking_failed"
        state.errors.append(
            "No paper text available for chunking."
        )
        return state

    try:
        state.chunks = chunk_text(state.paper_text)

        if not state.chunks:
            raise ValueError("No chunks were created.")

        state.status = "paper_chunked"

    except Exception as exc:
        state.errors.append(
            f"Failed to chunk paper: {exc}"
        )
        state.status = "chunking_failed"

    return state

def build_vector_store_node(state: AgentState) -> AgentState:
    """
    Build a FAISS vector store from the paper chunks.
    """

    if not state.chunks:
        state.status = "vector_store_failed"
        state.errors.append(
            "No chunks available for building the vector store."
        )
        return state

    try:
        vector_store = VectorStore()
        vector_store.build(state.chunks)

        state.vector_store = vector_store
        state.status = "vector_store_built"

    except Exception as exc:
        state.errors.append(
            f"Failed to build vector store: {exc}"
        )
        state.status = "vector_store_failed"

    return state


def retrieve_relevant_chunks_node(
    state: AgentState,
    question: str,
    top_k: int = 5,
) -> AgentState:
    """
    Retrieve the most relevant paper chunks for a question.
    """

    if state.vector_store is None:
        state.status = "chunk_retrieval_failed"
        state.errors.append(
            "Vector store is not available for retrieval."
        )
        return state

    if not question.strip():
        state.status = "chunk_retrieval_failed"
        state.errors.append(
            "Question cannot be empty."
        )
        return state

    try:
        results = state.vector_store.search(
            question,
            top_k=top_k,
        )

        state.retrieved_chunks = [
            result["chunk"]
            for result in results
        ]

        state.status = "chunks_retrieved"

    except Exception as exc:
        state.errors.append(
            f"Failed to retrieve relevant chunks: {exc}"
        )
        state.status = "chunk_retrieval_failed"

    return state   

def generate_briefing_node(state: AgentState) -> AgentState:
    """
    Generate an executive briefing using relevant paper chunks
    retrieved from the vector store.
    """

    if not state.selected_paper:
        state.status = "briefing_failed"
        state.errors.append(
            "No selected paper available for briefing."
        )
        return state

    if state.vector_store is None:
        state.status = "briefing_failed"
        state.errors.append(
            "Vector store is not available for briefing."
        )
        return state

    try:
        from llm.prompts import build_briefing_prompt
        from llm.client import generate_response

        state.paper_metadata = state.selected_paper

        retrieval_questions = [
            "What problem does this paper address and why is the problem important?",
            "What method or approach does the paper propose?",
            "What are the key results, experiments, findings, or claims reported by the paper?",
            "What limitations, weaknesses, assumptions, or future work does the paper explicitly discuss?",
        ]

        retrieved_chunks = []
        seen_chunk_ids = set()

        for question in retrieval_questions:
            results = state.vector_store.search(
                question,
                top_k=2,
            )

            for result in results:
                chunk = result["chunk"]
                chunk_id = chunk.get("id")

                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    retrieved_chunks.append(chunk)

        if not retrieved_chunks:
            state.status = "briefing_failed"
            state.errors.append(
                "No relevant paper chunks were retrieved for briefing."
            )
            return state

        # Keep the context bounded for the LLM request.
        retrieved_chunks = retrieved_chunks[:8]

        briefing_text = "\n\n".join(
            chunk["text"] for chunk in retrieved_chunks
        )

        prompt = build_briefing_prompt(
            state.paper_metadata,
            briefing_text,
        )

        response = generate_response(prompt)
        if not response or not response.strip():
            raise ValueError("LLM returned an empty briefing.")

        state.briefing = {
            "content": response,
        }

        state.status = "briefing_generated"

    except Exception as exc:
        state.errors.append(
            f"Failed to generate briefing: {exc}"
        )
        state.status = "briefing_failed"

    return state

def qa_node(
    state: AgentState,
    question: str,
) -> AgentState:
    """
    Answer a user question using retrieved paper chunks.
    """

    if state.vector_store is None:
        state.status = "qa_failed"
        state.errors.append(
            "Vector store is not available for QA."
        )
        return state

    if not question.strip():
        state.status = "qa_failed"
        state.errors.append(
            "Question cannot be empty."
        )
        return state

    try:
        from llm.prompts import build_qa_prompt
        from llm.client import generate_response

        # Retrieve evidence for the current question.
        state = retrieve_relevant_chunks_node(
            state,
            question,
            top_k=5,
        )

        if state.status != "chunks_retrieved":
            state.status = "qa_failed"
            return state

        prompt = build_qa_prompt(
            question,
            state.retrieved_chunks,
            state.conversation_history,
        )

        answer = generate_response(prompt)

        state.conversation_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        state.conversation_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        state.status = "qa_answered"

    except Exception as exc:
        state.errors.append(
            f"Failed to answer question: {exc}"
        )
        state.status = "qa_failed"

    return state
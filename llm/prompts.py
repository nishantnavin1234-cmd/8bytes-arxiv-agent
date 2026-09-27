def build_briefing_prompt(paper_metadata: dict, paper_text: str) -> str:
    """
    Build a prompt for generating an executive paper briefing.
    """

    return f"""
You are an academic research assistant.

Create an executive briefing for the following arXiv paper.

PAPER METADATA:
Title: {paper_metadata.get("title", "")}
Authors: {", ".join(paper_metadata.get("authors", []))}
arXiv ID: {paper_metadata.get("id", "")}
Published: {paper_metadata.get("published", "")}
arXiv URL: {paper_metadata.get("arxiv_url", "")}

PAPER TEXT:
{paper_text}

Your briefing must contain:

1. Title
2. Authors
3. arXiv ID
4. Publish date
5. arXiv link
6. Why this paper matters
   - Explain this in plain English for a technically literate reader.
7. Problem
   - What problem does the paper address?
8. Method
   - Explain the main approach using concise bullet points.
9. Key results / claims
   - Report the important results or claims made by the paper.
10. Limitations
   - Explicitly state limitations mentioned by the paper.
   - Do not invent limitations.
11. Suggested follow-up questions
   - Provide useful questions a reader could ask about the paper.

GROUNDING RULES:
- Use only information supported by the provided paper text and metadata.
- Do not invent facts, results, experiments, limitations, or conclusions.
- If the paper does not provide enough information for a requested section, explicitly say that the information is not available in the provided paper.
- Clearly distinguish the paper's claims from your own explanation.
- Do not use outside sources.

Return the briefing in clear Markdown.
"""

def build_qa_prompt(
    question: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict],
) -> str:
    """
    Build a grounded prompt for answering a question about the paper.
    """

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )

    return f"""
You are an academic research assistant answering questions about an arXiv paper.

Answer the user's question using ONLY the retrieved paper chunks below.

RETRIEVED PAPER CHUNKS:
{context}

CONVERSATION HISTORY:
{history if history else "No previous conversation."}

USER QUESTION:
{question}

GROUNDING RULES:
- Use only information supported by the retrieved paper chunks.
- Do not use outside knowledge.
- Do not invent facts, results, methods, or conclusions.
- If the answer is not supported by the retrieved chunks, explicitly say:
  "The answer is not available in the retrieved paper content."
- If the paper makes a claim, describe it as the paper's claim.
- Answer clearly and concisely.
"""
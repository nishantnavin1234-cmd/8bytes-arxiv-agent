from agent.state import AgentState
from agent.nodes import (
    query_understanding_node,
    retrieve_papers_node,
    select_paper_node,
    fetch_and_parse_node,
    chunk_paper_node,
    build_vector_store_node,
    generate_briefing_node,
)


def run_graph(input_query: str) -> AgentState:
    """
    Run the main arXiv paper analysis workflow.
    """

    state = AgentState(input_query=input_query)

    state = query_understanding_node(state)

    if state.status != "query_understood":
        return state

    state = retrieve_papers_node(state)

    if state.status != "papers_retrieved":
        return state

    state = select_paper_node(state)

    if state.status != "paper_selected":
        return state

    state = fetch_and_parse_node(state)

    if state.status != "paper_parsed":
        return state

    state = chunk_paper_node(state)

    if state.status != "paper_chunked":
        return state

    state = build_vector_store_node(state)

    if state.status != "vector_store_built":
        return state

    state = generate_briefing_node(state)

    return state

def answer_question(
    state: AgentState,
    question: str,
) -> AgentState:
    """
    Answer a question using the paper state created by the main graph.
    """

    from agent.nodes import qa_node

    return qa_node(state, question)
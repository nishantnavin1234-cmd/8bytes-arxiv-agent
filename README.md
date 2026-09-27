# Autonomous arXiv Paper Digest & QA Agent

A stateful Python agent that retrieves arXiv papers from either a research topic or a specific arXiv ID/URL, parses the paper, builds a local vector index, generates a grounded executive briefing, and answers follow-up questions using retrieved paper content.

## Features

- Accepts a natural-language research topic or a specific arXiv paper ID/URL.
- Retrieves paper metadata using the official arXiv API.
- Selects a relevant paper for topic-based searches.
- Downloads and parses the paper PDF.
- Splits the paper into overlapping chunks.
- Removes the bibliography from the retrieval corpus.
- Creates local embeddings using Sentence Transformers.
- Stores embeddings in a local FAISS vector index.
- Generates an executive briefing using an LLM.
- Answers follow-up questions using retrieved paper chunks.
- Maintains conversation history in persistent agent state.
- Handles retrieval, parsing, chunking, vector-store, retrieval, QA, and LLM failures gracefully.

## Architecture

The agent is implemented as a stateful workflow. Each node receives the same `AgentState`, performs one focused operation, updates the state, and passes it to the next stage.

```text
User Input
    |
    v
Query Understanding
    |
    +---- Specific arXiv ID/URL ----+
    |                               |
    |                               v
    |                         Retrieve Paper
    |
    +---- Research Topic -----------+
                                    |
                                    v
                              Search arXiv
                                    |
                                    v
                              Select Paper
                                    |
                                    v
                            Fetch & Parse PDF
                                    |
                                    v
                               Chunk Paper
                                    |
                                    v
                           Build FAISS Store
                                    |
                                    v
                           Generate Briefing
                                    |
                                    v
                              Paper State
                                    |
                                    v
                              QA Question
                                    |
                                    v
                         Retrieve Relevant Chunks
                                    |
                                    v
                              LLM Answer
                                    |
                                    v
                         Update Conversation History
```

### State

`AgentState` is the shared state passed between workflow nodes.

It stores:

- User input and detected query type.
- Retrieved candidate papers.
- Selected paper and metadata.
- Parsed paper text.
- Paper chunks.
- Retrieved chunks for QA.
- The FAISS vector store.
- Generated executive briefing.
- Conversation history.
- Execution status and errors.

This allows later nodes, especially the QA loop, to reuse information produced by earlier nodes without rebuilding the entire workflow.

### Main Workflow Nodes

1. **Query Understanding**  
   Determines whether the input is a specific arXiv paper or a research topic.

2. **Paper Retrieval**  
   Retrieves the requested paper or searches arXiv for candidate papers.

3. **Paper Selection**  
   Selects the relevant paper from topic-search results using title and abstract term overlap.

4. **Fetch & Parse**  
   Downloads the selected PDF and extracts its text using PyMuPDF.

5. **Chunking**  
   Splits the paper into overlapping chunks and removes the bibliography section from the retrieval corpus.

6. **Vector Store**  
   Generates embeddings using Sentence Transformers and stores them in a local FAISS index.

7. **Briefing Generation**  
   Retrieves relevant paper content and uses the LLM to generate the executive briefing.

8. **QA Loop**  
   Retrieves chunks relevant to each question and generates an answer using the retrieved paper content. Each question and answer is stored in `conversation_history`.

### Failure Handling

Each major node checks its prerequisites and catches failures. Errors are stored in `AgentState.errors` and the workflow status is updated to indicate the failed stage.

Examples include:

- No arXiv results.
- PDF download or parsing failure.
- No chunks created.
- Vector-store construction failure.
- Missing vector store during QA.
- Empty QA questions.
- Empty LLM responses.

The workflow stops at the failed stage rather than continuing with invalid state.

## Project Structure

```text
8bytes-arxiv-agent/
│
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── nodes.py
│   ├── chunker.py
│   └── vector_store.py
│
├── arxiv_api/
│   ├── __init__.py
│   ├── client.py
│   └── pdf_parser.py
│
├── llm/
│   ├── __init__.py
│   ├── client.py
│   └── prompts.py
│
├── .env
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone or download the project

Open a terminal in the project directory.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git using `.gitignore`.

## Running the Agent

Run:

```bash
python main.py
```

The CLI accepts either a specific arXiv paper or a natural-language research topic.

### Specific arXiv paper

Example:

```text
2510.14973v2
```

The agent also accepts an arXiv abstract URL:

```text
https://arxiv.org/abs/2510.14973v2
```

or a PDF URL:

```text
https://arxiv.org/pdf/2510.14973v2.pdf
```

### Research topic

Example:

```text
KV cache compression for large language models
```

For a topic query, the agent searches arXiv, retrieves candidate papers, selects a relevant paper, and continues through the paper analysis pipeline.

## Executive Briefing

For a selected paper, the agent generates an executive briefing containing:

- Title
- Authors
- arXiv ID
- Publication date
- arXiv link
- Why the paper matters
- Problem being addressed
- Main method
- Key results or claims
- Limitations
- Suggested follow-up questions

The briefing prompt instructs the LLM to use only the supplied paper content and metadata and not invent unsupported information.

## Grounded Question Answering

After the paper has been processed, questions can be answered using the same `AgentState`.

The QA flow is:

```text
User Question
      |
      v
Vector Store Retrieval
      |
      v
Relevant Paper Chunks
      |
      v
Grounded LLM Prompt
      |
      v
Answer
      |
      v
Conversation History
```

The QA prompt instructs the LLM to use only the retrieved paper chunks. If the answer is not supported by the retrieved content, the agent is instructed to explicitly state that the answer is not available in the retrieved paper content.

The same state can then be used for follow-up questions, preserving the previous conversation.

## Example

### Input

```text
2510.14973v2
```

### Selected Paper

**Attention Is All You Need for KV Cache in Diffusion LLMs**

**Authors:** Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen

**arXiv ID:** 2510.14973v2

The agent retrieves the paper, downloads and parses the PDF, creates chunks and embeddings, builds the FAISS index, and generates the executive briefing.

### Example QA Exchange 1

**Question:**

```text
What is Elastic-Cache and what problem does it solve?
```

**Answer:**

The paper proposes Elastic-Cache, a training-free caching policy for diffusion LLMs. It uses changes in attention patterns to decide when cached key/value pairs should be refreshed, reducing unnecessary full-model computation while maintaining model performance.

### Example QA Exchange 2

**Question:**

```text
How does the paper decide when to refresh the cache?
```

**Answer:**

The method monitors attention patterns across decoding steps. It computes the similarity between the current and previous attention patterns and uses a threshold to determine when the cached values should be refreshed.

## Design Decisions and Trade-offs

### Custom stateful workflow

A custom Python state machine was used instead of introducing a larger orchestration framework. The assessment allows a custom state machine, and the implementation keeps the workflow explicit and easy to inspect.

### arXiv Python package

The `arxiv` Python package is used to communicate with the official arXiv API and retrieve paper metadata.

### PyMuPDF

PyMuPDF is used for PDF text extraction because the system needs the paper's actual text for chunking and retrieval.

### Sentence Transformers

`all-MiniLM-L6-v2` is used to create local sentence embeddings without requiring a paid embedding API.

### FAISS

FAISS provides a local vector index for semantic retrieval. This keeps the retrieval layer lightweight and avoids requiring an external vector database.

### Retrieval-grounded generation

The briefing and QA prompts receive retrieved paper content rather than relying only on the model's general knowledge. This reduces the risk of unsupported answers.

### Bibliography filtering

The bibliography is removed before chunking because reference entries can otherwise dominate semantic retrieval and lead QA toward citations instead of the paper's actual content.

### Local processing

Paper parsing, chunking, embedding, and vector search are performed locally. The LLM is used only for generation.

## Error Handling

The system tracks execution status and errors in `AgentState`.

Examples of handled failure cases include:

```text
retrieval_failed
selection_failed
parse_failed
chunking_failed
vector_store_failed
chunk_retrieval_failed
briefing_failed
qa_failed
```

For example, if the LLM returns an empty response, the system does not incorrectly mark the briefing as successful. It records the failure and reports:

```text
LLM returned an empty briefing.
```

## Technologies Used

- **Python**
- **arXiv API**
- **arxiv Python package**
- **PyMuPDF**
- **Sentence Transformers**
- **FAISS**
- **Groq API**
- **Pydantic**
- **python-dotenv**

## Example Workflow

```text
Input:
"KV cache compression for large language models"

        ↓

Query Understanding
        ↓
Topic detected
        ↓
Search arXiv
        ↓
Retrieve candidate papers
        ↓
Select relevant paper
        ↓
Download PDF
        ↓
Extract text
        ↓
Chunk paper
        ↓
Generate embeddings
        ↓
Build FAISS index
        ↓
Retrieve relevant content
        ↓
Generate executive briefing
        ↓
Answer follow-up questions
```

For a specific paper ID or URL, the search and selection stages are replaced by direct paper retrieval.

## Scope

The project focuses on arXiv research-paper retrieval, paper analysis, grounded briefing generation, and question answering.

It intentionally does not include:

- Authentication systems
- Deployment infrastructure
- Production monitoring
- Non-arXiv sources
- Fine-tuning
- A full web frontend

The project is designed as a lightweight CLI-based research assistant in accordance with the assessment scope.
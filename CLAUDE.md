# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Uses ChromaDB for vector storage, Anthropic Claude for AI generation, sentence-transformers for embeddings, and FastAPI for the backend.

## Commands

Always use `uv` to run the server, manage dependencies, and execute Python files. Do not use `pip` or bare `python` directly.

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# App available at http://localhost:8000
```

There are no tests, linter, or formatter configured in this project.

## Environment Setup

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=<key>`. See `.env.example`.

## Architecture

### Query Flow (two-call pattern)

Frontend POSTs to `/api/query` → FastAPI route → `RAGSystem.query()` → `AIGenerator.generate_response()` sends the query to Claude with a `search_course_content` tool available. If Claude decides to search, the tool executes against ChromaDB, results are fed back, and a **second Claude API call** (without tools) produces the final synthesized answer.

### Backend Components (`backend/`)

- **`app.py`** — FastAPI app. Two endpoints: `POST /api/query` (chat) and `GET /api/courses` (stats). Serves frontend as static files from `../frontend`. Loads documents from `../docs` on startup.
- **`rag_system.py`** — Orchestrator. Wires together all components. Entry point is `query(query, session_id)`.
- **`ai_generator.py`** — Claude API client. System prompt is a class constant (`SYSTEM_PROMPT`). Uses `tool_choice: auto`. Handles the two-call tool execution loop in `_handle_tool_execution()`.
- **`search_tools.py`** — `Tool` ABC + `CourseSearchTool` implementation + `ToolManager` registry. The search tool supports optional `course_name` and `lesson_number` filters. Tracks `last_sources` for the UI.
- **`vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (course titles for semantic name resolution) and `course_content` (chunked text). The `search()` method first resolves a fuzzy course name against the catalog, then queries content with filters.
- **`document_processor.py`** — Parses structured course documents (expected format: metadata header lines, then `Lesson N:` markers). Chunks text by sentences with configurable size/overlap. Prepends lesson/course context to chunks.
- **`session_manager.py`** — In-memory conversation history per session. Keeps last N exchanges (configurable via `MAX_HISTORY`).
- **`config.py`** — Dataclass with all settings. Model: `claude-sonnet-4-20250514`, embeddings: `all-MiniLM-L6-v2`, chunk size: 800 chars, overlap: 100, max results: 5, max history: 2 exchanges.
- **`models.py`** — Pydantic models: `Course`, `Lesson`, `CourseChunk`.

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS chat interface. Uses `marked.js` for markdown rendering. Session ID is tracked client-side and sent with each request.

### Document Format (`docs/`)

Course text files follow a specific structure:
```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: Introduction
Lesson Link: ...
[content]

Lesson 1: ...
[content]
```

### Key Design Decisions

- Course titles serve as unique IDs throughout the system (ChromaDB IDs, deduplication keys).
- Course name filtering uses semantic search against the catalog collection rather than exact matching, so partial/fuzzy names work.
- The tool execution strips tools from the second API call, so Claude cannot chain multiple searches.
- ChromaDB is persisted to `./chroma_db` (relative to backend dir). On startup, existing courses are skipped to avoid re-indexing.
- All state (sessions, vector store) is in-memory/local — no external database.

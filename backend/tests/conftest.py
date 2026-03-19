import pytest
from unittest.mock import MagicMock, Mock

from vector_store import SearchResults

# ── SearchResults Fixtures ──


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'Nonexistent Course'",
    )


@pytest.fixture
def single_result_search_results():
    return SearchResults(
        documents=["This is lesson content about transformers."],
        metadata=[
            {
                "course_title": "Introduction to AI",
                "lesson_number": 3,
                "chunk_index": 0,
            }
        ],
        distances=[0.25],
    )


@pytest.fixture
def multi_result_search_results():
    """Multiple docs; two from same (course, lesson) to test dedup."""
    return SearchResults(
        documents=[
            "Content about neural networks from lesson 1.",
            "More content about neural networks from lesson 1.",
            "Content about CNNs from lesson 2.",
        ],
        metadata=[
            {
                "course_title": "Deep Learning Fundamentals",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "Deep Learning Fundamentals",
                "lesson_number": 1,
                "chunk_index": 1,
            },
            {
                "course_title": "Deep Learning Fundamentals",
                "lesson_number": 2,
                "chunk_index": 5,
            },
        ],
        distances=[0.1, 0.15, 0.3],
    )


@pytest.fixture
def result_without_lesson_number():
    return SearchResults(
        documents=["General course overview content."],
        metadata=[
            {
                "course_title": "Python Basics",
                "lesson_number": None,
                "chunk_index": 0,
            }
        ],
        distances=[0.2],
    )


# ── Mock VectorStore ──


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    return store


# ── Mock RAG System ──


@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with sensible defaults."""
    rag = MagicMock()

    rag.query.return_value = (
        "This is the AI response.",
        [{"text": "AI Course - Lesson 1", "url": "https://example.com/lesson1"}],
    )

    rag.session_manager.create_session.return_value = "session_1"
    rag.session_manager.sessions = {}

    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to AI", "Deep Learning Fundamentals"],
    }

    return rag

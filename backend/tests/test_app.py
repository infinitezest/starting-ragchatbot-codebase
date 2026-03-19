import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ── Pydantic models (mirrors app.py, avoids importing it) ──


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ── Test app factory ──


def create_test_app(rag_system):
    """Build a FastAPI app with the same endpoints as app.py but using the given rag_system."""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        sessions = rag_system.session_manager.sessions
        if session_id in sessions:
            del sessions[session_id]
        return {"status": "ok"}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ── Fixtures ──


@pytest.fixture
def client(mock_rag_system):
    """TestClient wrapping the inline FastAPI app with a mock RAG system."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


# ── POST /api/query ──


class TestQueryEndpoint:

    def test_query_happy_path_with_session(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is AI?", "session_id": "s1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "This is the AI response."
        assert data["session_id"] == "s1"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "AI Course - Lesson 1"
        mock_rag_system.query.assert_called_once_with("What is AI?", "s1")

    def test_query_auto_creates_session_when_missing(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_auto_creates_session_when_null(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "Hello", "session_id": None})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "session_1"

    def test_query_returns_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Direct answer.", [])
        resp = client.post("/api/query", json={"query": "Hi", "session_id": "s1"})
        assert resp.json()["sources"] == []

    def test_query_returns_multiple_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Answer.",
            [
                {"text": "Course A - Lesson 1", "url": "https://a.com"},
                {"text": "Course B - Lesson 2", "url": None},
            ],
        )
        resp = client.post("/api/query", json={"query": "Q", "session_id": "s1"})
        sources = resp.json()["sources"]
        assert len(sources) == 2
        assert sources[1]["url"] is None

    def test_query_internal_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("ChromaDB connection failed")
        resp = client.post("/api/query", json={"query": "test", "session_id": "s1"})
        assert resp.status_code == 500
        assert "ChromaDB connection failed" in resp.json()["detail"]

    def test_query_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_query_empty_string_is_accepted(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200


# ── DELETE /api/session/{session_id} ──


class TestDeleteSessionEndpoint:

    def test_delete_existing_session(self, client, mock_rag_system):
        mock_rag_system.session_manager.sessions = {"s1": ["some", "history"]}
        resp = client.delete("/api/session/s1")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert "s1" not in mock_rag_system.session_manager.sessions

    def test_delete_nonexistent_session_still_ok(self, client, mock_rag_system):
        mock_rag_system.session_manager.sessions = {}
        resp = client.delete("/api/session/nonexistent")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_delete_does_not_affect_other_sessions(self, client, mock_rag_system):
        mock_rag_system.session_manager.sessions = {"s1": [], "s2": []}
        client.delete("/api/session/s1")
        assert "s2" in mock_rag_system.session_manager.sessions


# ── GET /api/courses ──


class TestCoursesEndpoint:

    def test_courses_happy_path(self, client, mock_rag_system):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 2
        assert "Introduction to AI" in data["course_titles"]

    def test_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_internal_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "DB error" in resp.json()["detail"]

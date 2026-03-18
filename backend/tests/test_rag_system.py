import pytest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "/tmp/test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-test"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def rag_system_with_mocks(mock_config):
    """RAGSystem with all heavy dependencies mocked."""
    with patch("rag_system.DocumentProcessor") as MockDP, \
         patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.SessionManager") as MockSM, \
         patch("rag_system.CourseSearchTool") as MockCST, \
         patch("rag_system.CourseOutlineTool") as MockCOT, \
         patch("rag_system.ToolManager") as MockTM:

        system = RAGSystem(mock_config)

        # Expose mock instances for assertions
        system._mock_ai = MockAI.return_value
        system._mock_session = MockSM.return_value
        system._mock_tool_manager = MockTM.return_value
        system._mock_vector_store = MockVS.return_value

        yield system


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() orchestration."""

    def test_query_returns_response_and_sources(self, rag_system_with_mocks):
        """query() returns (response_text, sources_list)."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "AI answer about neural nets."
        system._mock_tool_manager.get_last_sources.return_value = [
            {"text": "AI Course - Lesson 1", "url": "https://example.com"}
        ]
        system._mock_session.get_conversation_history.return_value = None

        response, sources = system.query("What are neural networks?", session_id="s1")

        assert response == "AI answer about neural nets."
        assert len(sources) == 1
        assert sources[0]["text"] == "AI Course - Lesson 1"

    def test_query_passes_wrapped_prompt(self, rag_system_with_mocks):
        """AI receives the query wrapped in the prompt template."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_session.get_conversation_history.return_value = None

        system.query("What is MCP?", session_id="s1")

        call_kwargs = system._mock_ai.generate_response.call_args[1]
        assert call_kwargs["query"] == "Answer this question about course materials: What is MCP?"

    def test_query_passes_tools_and_tool_manager(self, rag_system_with_mocks):
        """tools and tool_manager forwarded to AI generate_response."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_tool_manager.get_tool_definitions.return_value = [{"name": "search"}]
        system._mock_session.get_conversation_history.return_value = None

        system.query("Q", session_id="s1")

        call_kwargs = system._mock_ai.generate_response.call_args[1]
        assert call_kwargs["tools"] == [{"name": "search"}]
        assert call_kwargs["tool_manager"] is system._mock_tool_manager

    def test_query_with_session_passes_history(self, rag_system_with_mocks):
        """Conversation history fetched and forwarded when session exists."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_session.get_conversation_history.return_value = "User: Hi\nAssistant: Hello"

        system.query("Follow up", session_id="s1")

        call_kwargs = system._mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: Hi\nAssistant: Hello"
        system._mock_session.get_conversation_history.assert_called_once_with("s1")

    def test_query_without_session_passes_none_history(self, rag_system_with_mocks):
        """session_id=None means history is None."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []

        system.query("Q", session_id=None)

        call_kwargs = system._mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None

    def test_query_updates_session_history(self, rag_system_with_mocks):
        """add_exchange called with session_id, raw query, and response."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "The answer is 42."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_session.get_conversation_history.return_value = None

        system.query("What is the meaning?", session_id="s1")

        system._mock_session.add_exchange.assert_called_once_with(
            "s1", "What is the meaning?", "The answer is 42."
        )

    def test_query_skips_session_update_when_no_id(self, rag_system_with_mocks):
        """add_exchange not called when session_id is None."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []

        system.query("Q", session_id=None)

        system._mock_session.add_exchange.assert_not_called()

    def test_query_resets_sources(self, rag_system_with_mocks):
        """reset_sources() called after query completes."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_session.get_conversation_history.return_value = None

        system.query("Q", session_id="s1")

        system._mock_tool_manager.reset_sources.assert_called_once()

    def test_sources_reset_happens_after_get(self, rag_system_with_mocks):
        """get_last_sources is called BEFORE reset_sources."""
        system = rag_system_with_mocks
        call_order = []
        system._mock_tool_manager.get_last_sources.side_effect = lambda: (
            call_order.append("get"), [{"text": "src", "url": None}]
        )[1]
        system._mock_tool_manager.reset_sources.side_effect = lambda: call_order.append("reset")
        system._mock_ai.generate_response.return_value = "Answer."
        system._mock_session.get_conversation_history.return_value = None

        system.query("Q", session_id="s1")

        assert call_order == ["get", "reset"]

    def test_query_returns_empty_sources_when_no_search(self, rag_system_with_mocks):
        """Direct answer with no tool search returns empty sources."""
        system = rag_system_with_mocks
        system._mock_ai.generate_response.return_value = "Direct answer."
        system._mock_tool_manager.get_last_sources.return_value = []
        system._mock_session.get_conversation_history.return_value = None

        response, sources = system.query("General question", session_id="s1")

        assert sources == []

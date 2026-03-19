import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() output behavior."""

    def test_execute_with_results_formats_output(
        self, mock_vector_store, single_result_search_results
    ):
        """Formatted output contains [Course - Lesson N] header and content; last_sources populated."""
        mock_vector_store.search.return_value = single_result_search_results
        mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/ai/lesson3"
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="transformers")

        mock_vector_store.search.assert_called_once_with(
            query="transformers", course_name=None, lesson_number=None
        )
        assert "[Introduction to AI - Lesson 3]" in result
        assert "This is lesson content about transformers." in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Introduction to AI - Lesson 3"
        assert tool.last_sources[0]["url"] == "https://example.com/ai/lesson3"

    def test_execute_with_course_name_filter(
        self, mock_vector_store, single_result_search_results
    ):
        """course_name kwarg is forwarded to vector store search."""
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="attention", course_name="Introduction to AI")

        mock_vector_store.search.assert_called_once_with(
            query="attention", course_name="Introduction to AI", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(
        self, mock_vector_store, single_result_search_results
    ):
        """lesson_number kwarg is forwarded to vector store search."""
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="attention", lesson_number=5)

        mock_vector_store.search.assert_called_once_with(
            query="attention", course_name=None, lesson_number=5
        )

    def test_execute_with_both_filters(
        self, mock_vector_store, single_result_search_results
    ):
        """Both course_name and lesson_number forwarded together."""
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="attention", course_name="AI Course", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="attention", course_name="AI Course", lesson_number=2
        )

    def test_execute_empty_results_no_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Empty results without filters returns plain message."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum computing")

        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Empty results with course filter mentions the course name."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum", course_name="Physics 101")

        assert "in course 'Physics 101'" in result

    def test_execute_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Empty results with lesson filter mentions the lesson number."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum", lesson_number=7)

        assert "in lesson 7" in result

    def test_execute_empty_results_with_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Empty results with both filters mentions both."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum", course_name="Physics", lesson_number=3)

        assert "in course 'Physics'" in result
        assert "in lesson 3" in result

    def test_execute_with_error(self, mock_vector_store, error_search_results):
        """Error from SearchResults is returned directly."""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert result == "No course found matching 'Nonexistent Course'"

    def test_execute_deduplicates_sources(
        self, mock_vector_store, multi_result_search_results
    ):
        """Two chunks from same (course, lesson) produce only one source entry."""
        mock_vector_store.search.return_value = multi_result_search_results
        mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/dl/lesson1"
        )
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="neural networks")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Deep Learning Fundamentals - Lesson 1"
        assert tool.last_sources[1]["text"] == "Deep Learning Fundamentals - Lesson 2"

    def test_execute_result_without_lesson_number(
        self, mock_vector_store, result_without_lesson_number
    ):
        """No lesson_number → header is [Course] and get_course_link called."""
        mock_vector_store.search.return_value = result_without_lesson_number
        mock_vector_store.get_course_link.return_value = "https://example.com/python"
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="python basics")

        assert "[Python Basics]" in result
        assert "Lesson" not in result
        mock_vector_store.get_course_link.assert_called_once_with("Python Basics")
        mock_vector_store.get_lesson_link.assert_not_called()
        assert tool.last_sources[0]["url"] == "https://example.com/python"

    def test_execute_resolves_lesson_link(
        self, mock_vector_store, single_result_search_results
    ):
        """get_lesson_link called with correct course title and lesson number."""
        mock_vector_store.search.return_value = single_result_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/ai/L3"
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="transformers")

        mock_vector_store.get_lesson_link.assert_called_once_with(
            "Introduction to AI", 3
        )
        assert tool.last_sources[0]["url"] == "https://example.com/ai/L3"

    def test_execute_source_url_none_when_no_link(
        self, mock_vector_store, single_result_search_results
    ):
        """Source url is None when link lookup returns None."""
        mock_vector_store.search.return_value = single_result_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="transformers")

        assert tool.last_sources[0]["url"] is None


class TestToolManager:
    """Tests for ToolManager registration and dispatch."""

    def test_execute_tool_delegates_to_registered_tool(
        self, mock_vector_store, single_result_search_results
    ):
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "[Introduction to AI" in result

    def test_execute_unknown_tool_returns_error(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_last_sources_returns_tool_sources(
        self, mock_vector_store, single_result_search_results
    ):
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_all(
        self, mock_vector_store, single_result_search_results
    ):
        mock_vector_store.search.return_value = single_result_search_results
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        manager.reset_sources()

        assert manager.get_last_sources() == []
        assert tool.last_sources == []

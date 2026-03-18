import pytest
from unittest.mock import MagicMock, Mock, patch
from ai_generator import AIGenerator
from helpers import make_text_block, make_tool_use_block, make_api_response


class TestAIGeneratorDirectResponse:
    """Tests for direct (non-tool) responses."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_response_returns_text(self, MockAnthropic):
        """stop_reason='end_turn' returns response text directly."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Paris is the capital of France.")], stop_reason="end_turn"
        )

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(query="What is the capital of France?")

        assert result == "Paris is the capital of France."
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_response_api_params(self, MockAnthropic):
        """API called with correct model, temperature, max_tokens."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer.")], stop_reason="end_turn"
        )

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="Hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tools_means_no_tool_choice(self, MockAnthropic):
        """When tools=None, tools/tool_choice absent from API params."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer.")], stop_reason="end_turn"
        )

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_adds_tool_choice_auto(self, MockAnthropic):
        """When tools provided, tool_choice set to auto."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer.")], stop_reason="end_turn"
        )
        tools = [{"name": "search", "description": "Search", "input_schema": {}}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Hello", tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}


class TestAIGeneratorConversationHistory:
    """Tests for conversation history in system prompt."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_system_prompt_without_history(self, MockAnthropic):
        """No history: system prompt is the base SYSTEM_PROMPT."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer.")], stop_reason="end_turn"
        )

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    @patch("ai_generator.anthropic.Anthropic")
    def test_system_prompt_with_history(self, MockAnthropic):
        """History provided: system prompt includes 'Previous conversation:' and history text."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer.")], stop_reason="end_turn"
        )
        history = "User: Hi\nAssistant: Hello!"

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Follow up", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]


class TestAIGeneratorToolExecution:
    """Tests for the two-call tool execution pattern."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_calls_tool_manager(self, MockAnthropic):
        """stop_reason='tool_use' triggers tool_manager.execute_tool with correct args."""
        mock_client = MockAnthropic.return_value
        first_response = make_api_response(
            [
                make_text_block("Let me search for that."),
                make_tool_use_block("tool_123", "search_course_content", {"query": "neural networks"}),
            ],
            stop_reason="tool_use",
        )
        second_response = make_api_response(
            [make_text_block("Neural networks are computational models.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "[AI Course - Lesson 1]\nNeural network content."
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="What are neural networks?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="neural networks"
        )
        assert result == "Neural networks are computational models."

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_api_call_has_no_tools(self, MockAnthropic):
        """Follow-up API call after tool execution omits tools and tool_choice."""
        mock_client = MockAnthropic.return_value
        first_response = make_api_response(
            [make_tool_use_block("t1", "search_course_content", {"query": "test"})],
            stop_reason="tool_use",
        )
        second_response = make_api_response(
            [make_text_block("Final answer.")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Q", tools=tools, tool_manager=mock_tool_manager)

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_api_call_includes_tool_results(self, MockAnthropic):
        """Follow-up messages contain tool_result with correct tool_use_id and content."""
        mock_client = MockAnthropic.return_value
        first_response = make_api_response(
            [make_tool_use_block("t_abc", "search_course_content", {"query": "MCP"})],
            stop_reason="tool_use",
        )
        second_response = make_api_response(
            [make_text_block("MCP stands for...")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "MCP tool results"

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Messages: [user query, assistant tool_use, user tool_result]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        tool_result_content = messages[2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "t_abc"
        assert tool_result_content[0]["content"] == "MCP tool results"

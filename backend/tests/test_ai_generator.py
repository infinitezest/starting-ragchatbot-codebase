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
    """Tests for the single-round tool execution pattern."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_calls_tool_manager(self, MockAnthropic):
        """stop_reason='tool_use' triggers tool_manager.execute_tool with correct args."""
        mock_client = MockAnthropic.return_value
        first_response = make_api_response(
            [
                make_text_block("Let me search for that."),
                make_tool_use_block(
                    "tool_123", "search_course_content", {"query": "neural networks"}
                ),
            ],
            stop_reason="tool_use",
        )
        second_response = make_api_response(
            [make_text_block("Neural networks are computational models.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = (
            "[AI Course - Lesson 1]\nNeural network content."
        )
        tools = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]

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
    def test_single_round_intermediate_call_includes_tools(self, MockAnthropic):
        """After 1 tool round (within MAX_TOOL_ROUNDS), the next API call still includes tools."""
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
        tools = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert second_call_kwargs["tools"] == tools
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_round_includes_tool_results(self, MockAnthropic):
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


class TestAIGeneratorTwoToolRounds:
    """Tests for two sequential tool rounds."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_returns_final_text(self, MockAnthropic):
        """Two sequential tool rounds followed by a final synthesis response."""
        mock_client = MockAnthropic.return_value
        response1 = make_api_response(
            [make_tool_use_block("t1", "get_course_outline", {"course_name": "AI"})],
            stop_reason="tool_use",
        )
        response2 = make_api_response(
            [
                make_tool_use_block(
                    "t2", "search_course_content", {"query": "neural nets"}
                )
            ],
            stop_reason="tool_use",
        )
        response3 = make_api_response(
            [make_text_block("AI course lesson 4 covers neural nets.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.side_effect = [response1, response2, response3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline...",
            "Neural net content...",
        ]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "AI course lesson 4 covers neural nets."
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_final_call_strips_tools(self, MockAnthropic):
        """After MAX_TOOL_ROUNDS exhausted, the final API call has no tools."""
        mock_client = MockAnthropic.return_value
        response1 = make_api_response(
            [make_tool_use_block("t1", "get_course_outline", {"course_name": "AI"})],
            stop_reason="tool_use",
        )
        response2 = make_api_response(
            [make_tool_use_block("t2", "search_course_content", {"query": "topic"})],
            stop_reason="tool_use",
        )
        response3 = make_api_response(
            [make_text_block("Final.")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [response1, response2, response3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline", "content"]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        final_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_kwargs
        assert "tool_choice" not in final_call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_intermediate_call_has_tools(self, MockAnthropic):
        """The second API call (after first tool round) still includes tools."""
        mock_client = MockAnthropic.return_value
        response1 = make_api_response(
            [make_tool_use_block("t1", "get_course_outline", {"course_name": "AI"})],
            stop_reason="tool_use",
        )
        response2 = make_api_response(
            [make_tool_use_block("t2", "search_course_content", {"query": "topic"})],
            stop_reason="tool_use",
        )
        response3 = make_api_response(
            [make_text_block("Final.")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [response1, response2, response3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline", "content"]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert second_call_kwargs["tools"] == tools
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_message_accumulation(self, MockAnthropic):
        """Final API call contains full conversation context from both rounds."""
        mock_client = MockAnthropic.return_value
        response1 = make_api_response(
            [make_tool_use_block("t1", "get_course_outline", {"course_name": "AI"})],
            stop_reason="tool_use",
        )
        response2 = make_api_response(
            [make_tool_use_block("t2", "search_course_content", {"query": "topic"})],
            stop_reason="tool_use",
        )
        response3 = make_api_response(
            [make_text_block("Final.")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [response1, response2, response3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline_result", "search_result"]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        final_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = final_call_kwargs["messages"]
        # user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["tool_use_id"] == "t1"
        assert messages[2]["content"][0]["content"] == "outline_result"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert messages[4]["content"][0]["tool_use_id"] == "t2"
        assert messages[4]["content"][0]["content"] == "search_result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_tool_execution_order(self, MockAnthropic):
        """Tools are executed in the correct order across rounds."""
        mock_client = MockAnthropic.return_value
        response1 = make_api_response(
            [make_tool_use_block("t1", "get_course_outline", {"course_name": "AI"})],
            stop_reason="tool_use",
        )
        response2 = make_api_response(
            [
                make_tool_use_block(
                    "t2", "search_course_content", {"query": "neural nets"}
                )
            ],
            stop_reason="tool_use",
        )
        response3 = make_api_response(
            [make_text_block("Final.")], stop_reason="end_turn"
        )
        mock_client.messages.create.side_effect = [response1, response2, response3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline", "content"]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0] == (("get_course_outline",), {"course_name": "AI"})
        assert calls[1] == (("search_course_content",), {"query": "neural nets"})


class TestAIGeneratorEarlyTermination:
    """Tests for early termination scenarios."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_use_returns_immediately(self, MockAnthropic):
        """When Claude doesn't request tools, returns after 1 API call."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Direct answer.")], stop_reason="end_turn"
        )

        mock_tool_manager = MagicMock()
        tools = [{"name": "search_course_content"}]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Q", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Direct answer."
        assert mock_client.messages.create.call_count == 1
        mock_tool_manager.execute_tool.assert_not_called()

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_without_tool_manager_returns_text(self, MockAnthropic):
        """When tool_manager is None, returns text even if stop_reason is tool_use."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = make_api_response(
            [
                make_text_block("I would search but can't."),
                make_tool_use_block("t1", "search_course_content", {"query": "test"}),
            ],
            stop_reason="tool_use",
        )

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Q",
            tools=[{"name": "search_course_content"}],
            tool_manager=None,
        )

        assert result == "I would search but can't."
        assert mock_client.messages.create.call_count == 1

    def test_max_tool_rounds_constant(self):
        """MAX_TOOL_ROUNDS is set to 2."""
        assert AIGenerator.MAX_TOOL_ROUNDS == 2

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

MODEL = "llama3.1"
BASE_URL = "http://localhost:11434/v1"


def text_response(text):
    """Build a mock OpenAI chat completion returning plain text (no tools)."""
    message = Mock()
    message.content = text
    message.tool_calls = None
    response = Mock()
    response.choices = [Mock(message=message)]
    return response


def tool_response(tool_calls, content=None):
    """Build a mock OpenAI chat completion requesting tool calls.

    tool_calls: list of (id, name, arguments_json) tuples.
    """
    message = Mock()
    message.content = content
    message.tool_calls = []
    for call_id, name, arguments in tool_calls:
        tc = Mock()
        tc.id = call_id
        tc.function = Mock()
        tc.function.name = name
        tc.function.arguments = arguments
        message.tool_calls.append(tc)
    response = Mock()
    response.choices = [Mock(message=message)]
    return response


class TestAIGenerator:
    """Test cases for AIGenerator (Ollama via OpenAI-compatible API)"""

    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator(BASE_URL, MODEL)

        assert generator.model == MODEL
        assert generator.base_params["model"] == MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = text_response(
                "This is a test response from the model."
            )
            mock_openai.return_value = mock_client

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response("What is AI?")

            assert response == "This is a test response from the model."

            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args[1]

            assert call_args["model"] == MODEL
            assert call_args["temperature"] == 0
            assert call_args["max_tokens"] == 800
            # System prompt is the first message, user query second
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][1] == {"role": "user", "content": "What is AI?"}
            assert "tools" not in call_args

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = text_response(
                "This is a test response from the model."
            )
            mock_openai.return_value = mock_client

            generator = AIGenerator(BASE_URL, MODEL)

            history = "Previous conversation context"
            response = generator.generate_response(
                "Follow up question", conversation_history=history
            )

            assert response == "This is a test response from the model."

            # Verify the system message includes history
            call_args = mock_client.chat.completions.create.call_args[1]
            assert "Previous conversation context" in call_args["messages"][0]["content"]

    def test_generate_response_with_tools_no_tool_use(self, mock_tool_manager):
        """Test response generation with tools available but not used"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = text_response(
                "This is a test response from the model."
            )
            mock_openai.return_value = mock_client

            generator = AIGenerator(BASE_URL, MODEL)

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "What is machine learning?", tools=tools, tool_manager=mock_tool_manager
            )

            assert response == "This is a test response from the model."

            # Verify tools were converted to OpenAI format and passed
            call_args = mock_client.chat.completions.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tool_choice"] == "auto"
            assert call_args["tools"][0]["type"] == "function"
            assert call_args["tools"][0]["function"]["name"] == "search_course_content"
            # input_schema becomes the function parameters
            assert "parameters" in call_args["tools"][0]["function"]

    def test_generate_response_with_tool_use(self, mock_tool_manager):
        """Test response generation when the model requests tool use"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_123", "search_course_content", '{"query": "test query"}')]
                ),
                text_response("Based on the search results, here is the answer."),
            ]

            generator = AIGenerator(BASE_URL, MODEL)

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                "What is in lesson 1?", tools=tools, tool_manager=mock_tool_manager
            )

            assert response == "Based on the search results, here is the answer."

            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="test query"
            )

            assert mock_client.chat.completions.create.call_count == 2

    def test_generate_response_tool_use_multiple_tools(self, mock_tool_manager):
        """Test response generation when the model requests multiple tools"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [
                        ("tool_123", "search_course_content", '{"query": "first query"}'),
                        (
                            "tool_456",
                            "get_course_outline",
                            '{"course_name": "Test Course"}',
                        ),
                    ]
                ),
                text_response("Here's the comprehensive answer."),
            ]

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Tell me about the course",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert response == "Here's the comprehensive answer."

            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content", query="first query"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline", course_name="Test Course"
            )

    def test_handle_tool_execution_conversation_flow(self, mock_tool_manager):
        """Test that tool execution properly maintains conversation flow"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_123", "search_course_content", '{"query": "lesson content"}')]
                ),
                text_response("Final answer with tool results."),
            ]

            generator = AIGenerator(BASE_URL, MODEL)

            generator.generate_response(
                "What's in lesson 1?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Inspect the message history sent on the final call
            final_call_args = mock_client.chat.completions.create.call_args_list[1][1]
            messages = final_call_args["messages"]

            # system, user, assistant (tool_calls), tool result
            assert len(messages) == 4
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What's in lesson 1?"
            assert messages[2]["role"] == "assistant"
            assert messages[2]["tool_calls"][0]["id"] == "tool_123"
            assert messages[3]["role"] == "tool"
            assert messages[3]["tool_call_id"] == "tool_123"
            assert messages[3]["content"] == "Mock search result"

    def test_generate_response_tool_execution_error(self, mock_tool_manager):
        """Test handling when a tool returns an error string (not an exception)"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_123", "search_course_content", '{"query": "test"}')]
                ),
                text_response("I encountered an error searching."),
            ]

            mock_tool_manager.execute_tool.return_value = "Error: Tool execution failed"

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Search for something",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert response == "I encountered an error searching."

            final_call_args = mock_client.chat.completions.create.call_args_list[1][1]
            tool_result = final_call_args["messages"][-1]
            assert tool_result["content"] == "Error: Tool execution failed"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance"""
        assert "course materials and educational content" in AIGenerator.SYSTEM_PROMPT
        assert "Content Search Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Course Outline Tool" in AIGenerator.SYSTEM_PROMPT
        assert "You can make up to 2 rounds of tool calls" in AIGenerator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in AIGenerator.SYSTEM_PROMPT

    def test_api_parameters_consistency(self):
        """Test that API parameters are consistent across calls"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = text_response(
                "This is a test response from the model."
            )
            mock_openai.return_value = mock_client

            generator = AIGenerator(BASE_URL, MODEL)

            generator.generate_response("First question")
            generator.generate_response(
                "Second question", conversation_history="Previous context"
            )

            calls = mock_client.chat.completions.create.call_args_list
            for call in calls:
                args = call[1]
                assert args["model"] == MODEL
                assert args["temperature"] == 0
                assert args["max_tokens"] == 800

    def test_no_tool_manager_with_tool_use(self):
        """Test behavior when tools are requested but no tool_manager provided"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.return_value = tool_response(
                [("tool_1", "search_course_content", "{}")],
                content="I need to use a tool",
            )

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Search for something", tools=[{"name": "test_tool"}]
            )

            # Without a tool_manager, the message content is returned directly
            assert response == "I need to use a tool"
            assert mock_client.chat.completions.create.call_count == 1

    def test_no_tool_calls_returns_content_directly(self, mock_tool_manager):
        """A message with no tool_calls returns its content in a single call"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.return_value = text_response(
                "No tools were actually used."
            )

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert response == "No tools were actually used."
            mock_tool_manager.execute_tool.assert_not_called()
            assert mock_client.chat.completions.create.call_count == 1

    def test_sequential_tool_calling_two_rounds(self, mock_tool_manager):
        """Test sequential tool calling over 2 rounds"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_1", "get_course_outline", '{"course_name": "Test Course"}')]
                ),
                tool_response(
                    [("tool_2", "search_course_content", '{"query": "lesson 4 content"}')]
                ),
                text_response(
                    "Based on the course outline and lesson content, here's the answer."
                ),
            ]

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Find lesson 4 content from Test Course",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert (
                response
                == "Based on the course outline and lesson content, here's the answer."
            )

            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline", course_name="Test Course"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content", query="lesson 4 content"
            )

            assert mock_client.chat.completions.create.call_count == 3

    def test_sequential_tool_calling_early_termination(self, mock_tool_manager):
        """Test that sequential tool calling terminates early without tool use"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_1", "search_course_content", '{"query": "test query"}')]
                ),
                text_response("Here's the direct answer based on the search results."),
            ]

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "What is in the course?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert response == "Here's the direct answer based on the search results."

            assert mock_tool_manager.execute_tool.call_count == 1
            mock_tool_manager.execute_tool.assert_called_with(
                "search_course_content", query="test query"
            )

            assert mock_client.chat.completions.create.call_count == 2

    def test_sequential_tool_calling_tool_failure_stops_rounds(self, mock_tool_manager):
        """Test that a tool execution exception stops sequential rounds"""
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_client.chat.completions.create.side_effect = [
                tool_response(
                    [("tool_1", "search_course_content", '{"query": "test query"}')]
                ),
                text_response("I encountered an error while searching."),
            ]

            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

            generator = AIGenerator(BASE_URL, MODEL)

            response = generator.generate_response(
                "Search for something",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            assert response == "I encountered an error while searching."

            assert mock_tool_manager.execute_tool.call_count == 1
            assert mock_client.chat.completions.create.call_count == 2

            # The error is captured in the tool result message
            final_call_args = mock_client.chat.completions.create.call_args_list[1][1]
            tool_result_message = final_call_args["messages"][-1]
            assert tool_result_message["role"] == "tool"
            assert "Error: Tool execution failed" in tool_result_message["content"]

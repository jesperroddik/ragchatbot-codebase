import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


class AIGenerator:
    """Handles interactions with a local LLM served by Ollama via its
    OpenAI-compatible API for generating responses."""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- Use content search for detailed questions about specific topics or lessons
- Use course outline tool for questions about course structure, lesson titles, or complete course overviews
- **You can make up to 2 rounds of tool calls to gather comprehensive information**
- Use multiple rounds for complex queries that require information gathering then refinement
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Course Outline Responses:
When using the course outline tool, always include:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles
- Present information in a clear, structured format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        # Ollama exposes an OpenAI-compatible endpoint; api_key is unused by
        # Ollama but the OpenAI client requires a non-empty string.
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential rounds of tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (Anthropic-style definitions)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # OpenAI-style chat format carries the system prompt as the first message
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        # Convert tool definitions to OpenAI function-calling format once
        openai_tools = self._convert_tools(tools) if tools else None

        # Execute up to 2 rounds of tool calling
        for _round_num in range(2):
            api_params = {**self.base_params, "messages": messages}

            # Add tools if available
            if openai_tools:
                api_params["tools"] = openai_tools
                api_params["tool_choice"] = "auto"

            # Get response from the local model
            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message

            # Handle tool execution if needed
            if message.tool_calls and tool_manager:
                messages, should_continue = self._handle_tool_execution(
                    message, messages, tool_manager
                )
                if not should_continue:
                    break
            else:
                # No tool use, return direct response
                return message.content

        # After max rounds (or a tool failure), make a final call without tools
        final_params = {**self.base_params, "messages": messages}
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-style tool definitions (with 'input_schema') into the
        OpenAI function-calling format that Ollama expects."""
        converted = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return converted

    def _handle_tool_execution(self, message, messages: List, tool_manager):
        """
        Execute the tool calls requested in `message` and append both the
        assistant request and the tool results to the message history.

        Args:
            message: The assistant message containing tool_calls
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, should_continue). should_continue is
            False if any tool execution failed, so the caller stops looping.
        """
        # Re-append the assistant message that requested the tool call(s). Every
        # tool_call below must be answered with a matching "tool" message, or the
        # next API call will be rejected.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        any_failed = False
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            try:
                tool_result = tool_manager.execute_tool(tool_name, **tool_args)
            except Exception as e:
                tool_result = f"Error: Tool execution failed - {str(e)}"
                any_failed = True

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )

        # Continue to the next round only if every tool succeeded
        return messages, not any_failed

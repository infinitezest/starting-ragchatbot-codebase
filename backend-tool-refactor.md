Refactor @backend/ai_generator.py to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds.

Current Behavior:
- Claude makes 1 tool call -> tools are removed from API params -> final response.
- If Claude wants another tool call after seeing results, it can't (gets empty response)

Desired Behavior:
- Each tool call should be a separate API request where Claude can reason about previous results.
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when information from different courses/lessons is needed.

Example Flow:
1. User: "Search for a course that discusses the same topic as lesson 4 of Course X"
2. Claude: Get course outline for Course X -> Gets title of Lesson 4.
3. Claude: Uses the title to search for a course that discusses the same topic -> returns course information.
4. Claude: Provides complete answer.

Requirements:
- Maximum 2 sequential rounds per user query.
- Terminate when: (a) 2 rounds completed, (b) Claude's response has no tool_use blocks, or (c) tool call fails.
- Preserve conversation context between rounds
- Handle tool execution errors gracefully

Notes:
- Update the system prompt in @backend/ai_generator.py
- Update the test @backend/tests/test_ai_generator.py
- Write tests that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details.

Use two parallel subagents to brainstorm possible plans. Do not implement any code.

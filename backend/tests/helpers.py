"""Shared mock helpers for Anthropic API responses."""
from unittest.mock import Mock


def make_text_block(text):
    block = Mock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_id, name, input_dict):
    block = Mock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def make_api_response(content_blocks, stop_reason="end_turn"):
    response = Mock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    return response

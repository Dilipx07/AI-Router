"""
AI Router v1
Canonical request and tool-call identifiers.
"""

from __future__ import annotations

import uuid


def create_request_id() -> str:
    """
    Create a unique identifier for one logical router request.
    """
    return uuid.uuid4().hex


def create_tool_call_id(
    request_id: str,
    index: int,
) -> str:
    """
    Create a canonical tool-call ID scoped to one request.
    """
    return f"call_{request_id}_{index}"

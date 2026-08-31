"""
AI Router v1
Token cost calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from providers.base import ProviderUsage


# ============================================================
# PRICING
# ============================================================

@dataclass(frozen=True)
class ModelPricing:
    """
    Token pricing for one provider/model.

    Prices are expressed as currency units per one million
    tokens.
    """

    input_per_million: float = 0.0

    output_per_million: float = 0.0


# ============================================================
# COST RESULT
# ============================================================

@dataclass(frozen=True)
class UsageCost:
    """
    Calculated cost for one usage record.
    """

    input_cost: float = 0.0

    output_cost: float = 0.0

    total_cost: float = 0.0


# ============================================================
# COST CALCULATION
# ============================================================

def calculate_cost(
    usage: ProviderUsage,
    pricing: ModelPricing,
) -> UsageCost:
    """
    Calculate estimated cost from normalized token usage.
    """

    normalized = usage.normalized()

    input_cost = (
        normalized.prompt_tokens
        / 1_000_000
        * pricing.input_per_million
    )

    output_cost = (
        normalized.completion_tokens
        / 1_000_000
        * pricing.output_per_million
    )

    return UsageCost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


# ============================================================
# SAFE COST CALCULATION
# ============================================================

def calculate_cost_optional(
    usage: ProviderUsage,
    pricing: Optional[ModelPricing],
) -> UsageCost:
    """
    Calculate cost when pricing is available.

    Missing pricing produces zero cost rather than failing
    the request.
    """

    if pricing is None:
        return UsageCost()

    return calculate_cost(
        usage,
        pricing,
    )

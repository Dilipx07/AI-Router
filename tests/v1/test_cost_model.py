from providers.base import ProviderUsage
from router.cost import (
    ModelPricing,
    UsageCost,
    calculate_cost,
    calculate_cost_optional,
)


usage = ProviderUsage(
    prompt_tokens=100_000,
    completion_tokens=20_000,
    total_tokens=120_000,
)

pricing = ModelPricing(
    input_per_million=1.00,
    output_per_million=2.00,
)

cost = calculate_cost(
    usage,
    pricing,
)

print("=== COST MODEL TEST ===")
print("input_cost:", cost.input_cost)
print("output_cost:", cost.output_cost)
print("total_cost:", cost.total_cost)

assert cost.input_cost == 0.1
assert cost.output_cost == 0.04
assert cost.total_cost == 0.14


# ------------------------------------------------------------
# Zero usage
# ------------------------------------------------------------

zero = calculate_cost(
    ProviderUsage(),
    pricing,
)

assert zero == UsageCost()


# ------------------------------------------------------------
# Missing pricing
# ------------------------------------------------------------

missing = calculate_cost_optional(
    usage,
    None,
)

assert missing == UsageCost()


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

derived = calculate_cost(
    ProviderUsage(
        prompt_tokens=10_000,
        completion_tokens=5_000,
        total_tokens=0,
    ),
    ModelPricing(
        input_per_million=1.0,
        output_per_million=1.0,
    ),
)

assert derived.total_cost == 0.015


print()
print("input pricing: PASS")
print("output pricing: PASS")
print("total pricing: PASS")
print("missing pricing: PASS")
print("usage normalization: PASS")
print()
print("RESULT: PASS")

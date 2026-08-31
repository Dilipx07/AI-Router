from router.engine import RouterEngine, RoutingRequest
from router.models import ModelCandidate
from router.health import HealthStore


class TestRegistry:
    def __init__(self):
        self.candidates = [
            ModelCandidate(
                provider="ollama",
                model="test-no-tools",
                priority=100,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="test-with-tools",
                priority=90,
                capabilities=frozenset({"chat", "tools"}),
            ),
        ]

    def all(self, enabled_only=True):
        return list(self.candidates)


class TestHealth:
    def is_available(self, provider, model):
        return True

    def get(self, provider, model):
        raise AssertionError(
            "Health record should not be requested"
        )


engine = RouterEngine(
    model_registry=TestRegistry(),
    health_store=TestHealth(),
)

request = RoutingRequest(
    required_capabilities=frozenset({"tools"}),
    provider="ollama",
)

decision = engine.route(request)

print("=== TOOL CAPABILITY FILTER TEST ===")
print("success:", decision.success)
print(
    "selected:",
    decision.candidate.identifier
    if decision.candidate
    else None,
)
print("considered:", decision.candidates_considered)
print("available:", decision.candidates_available)
print("compatible:", decision.candidates_compatible)
print(
    "ranked:",
    [
        x.candidate.identifier
        for x in decision.ranked
    ],
)

print()
print("rejected:")

for item in decision.rejected:
    print(item)

no_tools_rejected = any(
    item.get("candidate") == "ollama/test-no-tools"
    and item.get("reason") == "missing_capabilities"
    and "tools" in item.get("missing", [])
    for item in decision.rejected
)

selected_tools_model = (
    decision.candidate is not None
    and decision.candidate.identifier
    == "ollama/test-with-tools"
)

print()
print(
    "no-tools model rejected:",
    no_tools_rejected,
)

print(
    "tool model selected:",
    selected_tools_model,
)

if not no_tools_rejected:
    raise AssertionError(
        "Model without tools was not rejected"
    )

if not selected_tools_model:
    raise AssertionError(
        "Tool-capable model was not selected"
    )

print()
print("RESULT: PASS")

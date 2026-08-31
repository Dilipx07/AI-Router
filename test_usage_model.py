from providers.base import ProviderUsage


def check(prompt, completion, total):
    usage = ProviderUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )

    normalized = usage.normalized()

    print(
        "input:",
        prompt,
        completion,
        total,
        "=>",
        normalized,
        "known=",
        normalized.known,
    )

    return normalized


a = check(10, 5, 15)

assert a.prompt_tokens == 10
assert a.completion_tokens == 5
assert a.total_tokens == 15
assert a.known


b = check(10, 5, 0)

assert b.total_tokens == 15
assert b.known


c = check(0, 0, 0)

assert c.prompt_tokens == 0
assert c.completion_tokens == 0
assert c.total_tokens == 0
assert not c.known


d = check(-10, -5, -1)

assert d.prompt_tokens == 0
assert d.completion_tokens == 0
assert d.total_tokens == 0


print()
print("RESULT: PASS")

from providers.openai_compatible import OpenAICompatibleProvider


cases = [
    {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    },
    {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
    },
    {
        "usage": {},
    },
    {},
    {
        "usage": None,
    },
]


for index, response in enumerate(cases, start=1):

    usage = OpenAICompatibleProvider._extract_usage(
        response
    )

    print(
        f"case {index}:",
        usage,
    )


a = OpenAICompatibleProvider._extract_usage(
    {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    }
)

assert a.prompt_tokens == 10
assert a.completion_tokens == 5
assert a.total_tokens == 15


b = OpenAICompatibleProvider._extract_usage(
    {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
    }
)

assert b.total_tokens == 15


c = OpenAICompatibleProvider._extract_usage(
    {}
)

assert c.prompt_tokens == 0
assert c.completion_tokens == 0
assert c.total_tokens == 0
assert not c.known


print()
print("RESULT: PASS")

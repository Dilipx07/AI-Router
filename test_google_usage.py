from providers.google import GoogleProvider


cases = [
    {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        }
    },
    {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
        }
    },
    {
        "usageMetadata": {},
    },
    {},
    {
        "usageMetadata": None,
    },
]


for index, response in enumerate(
    cases,
    start=1,
):

    usage = GoogleProvider._extract_usage(
        response
    )

    print(
        f"case {index}:",
        usage,
    )


a = GoogleProvider._extract_usage(
    {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        }
    }
)

assert a.prompt_tokens == 10
assert a.completion_tokens == 5
assert a.total_tokens == 15
assert a.known


b = GoogleProvider._extract_usage(
    {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
        }
    }
)

assert b.total_tokens == 15


c = GoogleProvider._extract_usage(
    {}
)

assert c.prompt_tokens == 0
assert c.completion_tokens == 0
assert c.total_tokens == 0
assert not c.known


print()
print("RESULT: PASS")

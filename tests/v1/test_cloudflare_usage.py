from providers.cloudflare import CloudflareProvider


cases = [
    {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    },
    {
        "result": {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 7,
            }
        }
    },
    {
        "result": {
            "response": {
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 8,
                    "total_tokens": 38,
                }
            }
        }
    },
    {},
]


for index, response in enumerate(cases, start=1):

    usage = CloudflareProvider._extract_usage(
        response
    )

    print(f"case {index}:", usage)


a = CloudflareProvider._extract_usage(
    cases[0]
)

assert a.prompt_tokens == 10
assert a.completion_tokens == 5
assert a.total_tokens == 15


b = CloudflareProvider._extract_usage(
    cases[1]
)

assert b.prompt_tokens == 20
assert b.completion_tokens == 7
assert b.total_tokens == 27


c = CloudflareProvider._extract_usage(
    cases[2]
)

assert c.prompt_tokens == 30
assert c.completion_tokens == 8
assert c.total_tokens == 38


d = CloudflareProvider._extract_usage(
    cases[3]
)

assert not d.known


print()
print("RESULT: PASS")

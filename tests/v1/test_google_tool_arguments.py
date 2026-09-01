import json

from providers.google import GoogleProvider


cases = [
    {
        "name": "simple",
        "args": {
            "city": "Bengaluru",
        },
    },
    {
        "name": "nested",
        "args": {
            "location": {
                "city": "Bengaluru",
                "country": "India",
            },
            "units": "metric",
        },
    },
    {
        "name": "array",
        "args": {
            "cities": [
                "Bengaluru",
                "Mumbai",
                "Delhi",
            ],
        },
    },
    {
        "name": "boolean_null",
        "args": {
            "enabled": True,
            "value": None,
        },
    },
]


for case in cases:

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "test_tool",
                                "args": case["args"],
                            }
                        }
                    ]
                }
            }
        ]
    }

    calls = GoogleProvider._extract_tool_calls(
        response
    )

    assert len(calls) == 1

    arguments = calls[0]["function"]["arguments"]

    print(
        case["name"],
        "type=",
        type(arguments).__name__,
        "value=",
        arguments,
    )

    assert isinstance(
        arguments,
        str,
    ), (
        f"{case['name']}: arguments must be a JSON string"
    )

    parsed = json.loads(arguments)

    assert parsed == case["args"]


print()
print("RESULT: PASS")

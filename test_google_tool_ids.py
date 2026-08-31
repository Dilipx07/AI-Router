import json

from providers.google import GoogleProvider


request_id = "request_test_001"

response = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {
                                "city": "Bengaluru",
                            },
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_temperature",
                            "args": {
                                "city": "Bengaluru",
                            },
                        }
                    },
                ]
            }
        }
    ]
}

calls = GoogleProvider._extract_tool_calls(
    response,
    request_id,
)

print("calls:", calls)

assert len(calls) == 2

assert calls[0]["id"] == "call_request_test_001_0"
assert calls[1]["id"] == "call_request_test_001_1"

assert calls[0]["function"]["name"] == "get_weather"
assert calls[1]["function"]["name"] == "get_temperature"

assert json.loads(
    calls[0]["function"]["arguments"]
) == {
    "city": "Bengaluru",
}

assert json.loads(
    calls[1]["function"]["arguments"]
) == {
    "city": "Bengaluru",
}

print()
print("RESULT: PASS")

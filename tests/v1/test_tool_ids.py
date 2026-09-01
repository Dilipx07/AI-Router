from router.ids import create_request_id, create_tool_call_id

request_id = create_request_id()

print("request_id:", request_id)
print("tool_0:", create_tool_call_id(request_id, 0))
print("tool_1:", create_tool_call_id(request_id, 1))
print("tool_2:", create_tool_call_id(request_id, 2))

assert request_id
assert create_tool_call_id(request_id, 0) != create_tool_call_id(request_id, 1)

other_request_id = create_request_id()

assert request_id != other_request_id
assert (
    create_tool_call_id(request_id, 0)
    != create_tool_call_id(other_request_id, 0)
)

print("RESULT: PASS")

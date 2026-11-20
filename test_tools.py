
from gemini_tool_demo import execute_tool, tool_declarations

print("Starting local agent workflow...")

# Step 1: Simulate the first tool request
print("\nStep 1: Add 20 + 10")
first_result = execute_tool(
    "add_numbers",
    {"a": 20, "b": 10},
)
assert first_result == 30
print("First result:", first_result)

# Step 2: Pass the first result into the second tool
print("\nStep 2: Subtract 5 from the first result")
final_result = execute_tool(
    "subtract_numbers",
    {"a": first_result, "b": 5},
)
assert final_result == 25
print("Final result:", final_result)

# Step 3: Check that both tools are declared
declared_names = {tool["name"] for tool in tool_declarations}
assert declared_names == {"add_numbers", "subtract_numbers"}

print("\nSUCCESS: Complete local tool workflow passed!")
print("Final answer:", final_result)

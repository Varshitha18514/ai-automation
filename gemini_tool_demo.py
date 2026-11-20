
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


# Tool 1: Add two numbers
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    print(f"Addition tool called with a={a}, b={b}")
    result = a + b
    print(f"Addition result: {result}")
    return result


# Tool 2: Subtract two numbers
def subtract_numbers(a: int, b: int) -> int:
    """Subtract the second number from the first number."""
    print(f"Subtraction tool called with a={a}, b={b}")
    result = a - b
    print(f"Subtraction result: {result}")
    return result


# Register local Python functions
available_functions = {
    "add_numbers": add_numbers,
    "subtract_numbers": subtract_numbers,
}


# Execute a tool requested by Gemini
def execute_tool(function_name, arguments):
    if function_name not in available_functions:
        raise ValueError(f"Unknown function: {function_name}")

    return available_functions[function_name](**arguments)


# Tell Gemini which tools are available
tool_declarations = [
    {
        "name": "add_numbers",
        "description": "Add two numbers and return the result.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "INTEGER"},
                "b": {"type": "INTEGER"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "subtract_numbers",
        "description": "Subtract the second number from the first number.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "INTEGER"},
                "b": {"type": "INTEGER"},
            },
            "required": ["a", "b"],
        },
    },
]


# Run the Gemini demo only when this file is executed directly
def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Check your .env file."
        )

    client = genai.Client(api_key=api_key)

    chat = client.chats.create(
        model="gemini-3.6-flash",
        config={
            "tools": [
                {"function_declarations": tool_declarations}
            ]
        },
    )

    response = chat.send_message(
        "First calculate 20 plus 10 using the addition tool. "
        "Then subtract 5 from that result using the subtraction tool. "
        "Tell me the final answer."
    )

    # Manual tool-execution loop
    while response.function_calls:
        tool_response_parts = []

        for function_call in response.function_calls:
            print(f"\nGemini requested: {function_call.name}")
            print(f"Arguments: {function_call.args}")

            result = execute_tool(
                function_call.name,
                function_call.args,
            )

            print(f"Manual tool result: {result}")

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": result},
                )
            )

        # Send local Python results back to Gemini
        response = chat.send_message(tool_response_parts)

    print("\nGemini's final answer:")
    print(response.text or "Gemini returned no text response.")


# Importing this file will NOT call the Gemini API.
if __name__ == "__main__":
    main()

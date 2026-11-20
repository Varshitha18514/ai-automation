
import os
import httpx

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)


# ==========================================
# 1. Python tools
# ==========================================

def add_numbers(a: int, b: int) -> int:
    print(f"Addition tool called: {a} + {b}")
    return a + b


def subtract_numbers(a: int, b: int) -> int:
    print(f"Subtraction tool called: {a} - {b}")
    return a - b


available_functions = {
    "add_numbers": add_numbers,
    "subtract_numbers": subtract_numbers,
}


def execute_tool(function_name, arguments):
    if function_name not in available_functions:
        raise ValueError(f"Unknown function: {function_name}")

    return available_functions[function_name](**arguments)


# ==========================================
# 2. Gemini tool declarations
# ==========================================

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
        "description": "Subtract the second number from the first.",
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


# ==========================================
# 3. Identify temporary errors
# ==========================================

def is_transient_error(error):

    # Temporary network and connection errors
    if isinstance(
        error,
        (
            ConnectionError,
            TimeoutError,
            httpx.TransportError,
        ),
    ):
        return True

    # Temporary Gemini API errors
    if isinstance(error, APIError):
        status_code = getattr(error, "code", None)

        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            return False

        return status_code in (408, 429, 500, 502, 503, 504)

    # Do not retry other errors
    return False


# ==========================================
# 4. Retry configuration
# ==========================================

def log_retry(retry_state):
    print(
        f"\nTemporary error: "
        f"{retry_state.outcome.exception()}"
    )

    print(
        f"Retrying in "
        f"{retry_state.next_action.sleep} seconds..."
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=8,
    ),
    retry=retry_if_exception(is_transient_error),
    before_sleep=log_retry,
    reraise=True,
)
def send_message_with_retry(chat, message):

    print("\nSending request to Gemini...")

    return chat.send_message(message)


# ==========================================
# 5. Main application
# ==========================================

def main():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    try:
        client = genai.Client(api_key=api_key)

        chat = client.chats.create(
            model="gemini-3.6-flash",
            config={
                "tools": [
                    {
                        "function_declarations": tool_declarations
                    }
                ]
            },
        )

        response = send_message_with_retry(
            chat,
            "First calculate 20 plus 10 using the addition tool. "
            "Then subtract 5 from that result using the subtraction tool. "
            "Tell me the final answer.",
        )

        # ==================================
        # 6. Manual function-calling loop
        # ==================================

        while response.function_calls:

            tool_response_parts = []

            for function_call in response.function_calls:

                print(
                    f"\nGemini requested: {function_call.name}"
                )

                print(
                    f"Arguments: {function_call.args}"
                )

                result = execute_tool(
                    function_call.name,
                    function_call.args,
                )

                print(f"Python tool result: {result}")

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result},
                    )
                )

            # Send tool results back to Gemini with retries
            response = send_message_with_retry(
                chat,
                tool_response_parts,
            )

        print("\nGemini's final answer:")
        print(
            response.text or "Gemini returned no text response."
        )

    except Exception as error:

        print("\nGemini workflow failed.")

        if is_transient_error(error):
            print(
                "Temporary API/network error remained "
                "after all retry attempts."
            )
        else:
            print(
                "A non-retryable error occurred."
            )

        print(f"Error details: {error}")


if __name__ == "__main__":
    main()

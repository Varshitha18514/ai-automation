
import asyncio
import os

from dotenv import load_dotenv
from google import genai


async def generate_response(
    client,
    prompt,
    semaphore,
    rate_limit_lock,
    rate_state,
    task_number,
):
    async with semaphore:

        # Keep request starts at least 13 seconds apart.
        async with rate_limit_lock:
            loop = asyncio.get_running_loop()
            current_time = loop.time()

            wait_time = 13 - (current_time - rate_state["last_request"])

            if wait_time > 0:
                print(
                    f"Task {task_number}: "
                    f"Waiting {wait_time:.1f} seconds for rate limit..."
                )
                await asyncio.sleep(wait_time)

            rate_state["last_request"] = loop.time()

        print(f"Task {task_number}: Sending request...")

        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )

            result = response.text or "No response received."

            print(f"Task {task_number}: Completed!")

            return {
                "task": task_number,
                "response": result,
                "status": "Success",
            }

        except Exception as error:
            print(f"Task {task_number}: Failed - {error}")

            return {
                "task": task_number,
                "response": str(error),
                "status": "Failed",
            }


async def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)

    prompts = [
        "Explain artificial intelligence in one sentence.",
        "Explain machine learning in one sentence.",
        "Explain deep learning in one sentence.",
        "Explain Python programming in one sentence.",
        "Explain API integration in one sentence.",
        "Explain asynchronous programming in one sentence.",
        "Explain automation in one sentence.",
        "Explain cloud computing in one sentence.",
        "Explain a database in one sentence.",
        "Explain cybersecurity in one sentence.",
    ]

    semaphore = asyncio.Semaphore(3)
    rate_limit_lock = asyncio.Lock()

    rate_state = {
        "last_request": 0.0,
    }

    tasks = [
        generate_response(
            client,
            prompt,
            semaphore,
            rate_limit_lock,
            rate_state,
            index + 1,
        )
        for index, prompt in enumerate(prompts)
    ]

    print(f"Starting {len(tasks)} asynchronous tasks...\n")

    try:
        results = await asyncio.gather(*tasks)

        print("\n========== FINAL RESULTS ==========")

        for result in results:
            print(f"\nTask {result['task']}: {result['status']}")
            print(result["response"])

        successful = sum(
            result["status"] == "Success"
            for result in results
        )

        print("\n========== SUMMARY ==========")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")

    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())

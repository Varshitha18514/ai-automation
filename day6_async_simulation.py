
import asyncio
import time


async def process_task(task_number, semaphore):
    async with semaphore:
        print(f"Task {task_number}: Started")

        # Simulate an API request taking 2 seconds.
        await asyncio.sleep(2)

        print(f"Task {task_number}: Completed")

        return f"Result from task {task_number}"


async def main():
    total_tasks = 50

    # Allow up to 10 tasks to run simultaneously.
    semaphore = asyncio.Semaphore(10)

    tasks = [
        process_task(number, semaphore)
        for number in range(1, total_tasks + 1)
    ]

    print(f"Starting {total_tasks} asynchronous tasks...")
    start_time = time.perf_counter()

    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total tasks: {len(results)}")
    print(f"Completed: {len(results)}")
    print(f"Failed: 0")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    print("\nAll 50 tasks completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

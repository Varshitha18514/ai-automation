
import json
from datetime import datetime


# Example pricing per 1 million tokens.
# These are illustrative values, not live Gemini pricing.
INPUT_PRICE_PER_MILLION = 0.10
OUTPUT_PRICE_PER_MILLION = 0.40


def calculate_cost(input_tokens, output_tokens):
    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MILLION

    return input_cost + output_cost


def log_usage(model, prompt, input_tokens, output_tokens):
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt": prompt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 8),
    }

    with open("usage_log.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


def main():
    print("========== AI TOKEN COST TRACKER ==========\n")

    sample_requests = [
        {
            "prompt": "Explain artificial intelligence.",
            "input_tokens": 10,
            "output_tokens": 50,
        },
        {
            "prompt": "Explain machine learning.",
            "input_tokens": 12,
            "output_tokens": 60,
        },
        {
            "prompt": "Explain asynchronous programming.",
            "input_tokens": 15,
            "output_tokens": 70,
        },
    ]

    total_cost = 0

    for request in sample_requests:
        record = log_usage(
            model="Gemini example",
            prompt=request["prompt"],
            input_tokens=request["input_tokens"],
            output_tokens=request["output_tokens"],
        )

        total_cost += record["estimated_cost_usd"]

        print(f"Prompt: {record['prompt']}")
        print(f"Input tokens: {record['input_tokens']}")
        print(f"Output tokens: {record['output_tokens']}")
        print(f"Total tokens: {record['total_tokens']}")
        print(f"Estimated cost: ${record['estimated_cost_usd']:.8f}")
        print("-------------------------------------------")

    print(f"\nTotal estimated cost: ${total_cost:.8f}")
    print("Usage records saved to usage_log.jsonl")


if __name__ == "__main__":
    main()

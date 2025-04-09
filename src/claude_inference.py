"""Module for generating responses using Bedrock runtime."""
import random
import os
import json
import argparse
import concurrent.futures
import boto3
from prompt import *
from src.dataset import load_data, load_formatting_func


# Configuration
BEDROCK_RUNTIME = boto3.client('bedrock-runtime', region_name='us-east-1')
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
MAX_TOKENS = 4096
BATCH_SIZE = 5
SYSTEM_PROMPT = ""
MULTIPLE_TEMPLATES = False


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature=0.0):
    """Generates a message using the specified model.

    Args:
        bedrock_runtime: Bedrock runtime client.
        model_id: ID of the model to use.
        system_prompt: System prompt for the model.
        messages: List of message dictionaries.
        max_tokens: Maximum number of tokens to generate.
        temperature: Temperature for response generation.

    Returns:
        dict: The generated response.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
    })
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body


def query_api(example, fpath):
    """Queries the API and saves the response.

    Args:
        example: Input example dictionary.
        fpath: File path to save the response.
    """
    message = [{"role": "user", "content": example['prompt']}]
    response = generate_message(BEDROCK_RUNTIME, MODEL_ID, SYSTEM_PROMPT, message, MAX_TOKENS)
    if response:
        output = response.get('content', [{}])[0].get('text', 'Error or no response')
        example['output'] = output
        print(f"Completed {fpath}")
        with open(fpath, 'w') as f:
            json.dump(example, f, indent=4)


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lastfm-stage2")
    parser.add_argument("--model_name", type=str, default="claude-3-sonnet")
    parser.add_argument("--split", type=str, default='test')
    args = parser.parse_args()

    print("Arguments:")
    print('=' * 50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('=' * 50)

    dataset = load_data[args.task](split=args.split)
    formatting_func = load_formatting_func[args.task]

    output_dir = os.path.join("output", args.task, args.model_name.split("/")[-1], args.split)
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    fpaths = []
    for i, item in enumerate(dataset):
        if os.path.exists(f'{output_dir}/{i}.json'):
            continue
        temp = item.copy()
        temp['prompt'] = formatting_func(temp)['prompt']
        prompts.append(temp)
        fpaths.append(f'{output_dir}/{i}.json')

    # Shuffle prompts and fpaths simultaneously
    print(f"Total prompts: {len(prompts)}")
    if len(prompts) > 0:
        combined = list(zip(prompts, fpaths))
        random.shuffle(combined)
        prompts, fpaths = zip(*combined)
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        executor.map(query_api, prompts, fpaths)


if __name__ == "__main__":
    main()

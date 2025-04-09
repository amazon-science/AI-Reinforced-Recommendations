import random
import argparse
import os
import boto3
import json
import concurrent.futures
from src.dataset import load_data, load_formatting_func

# Configuration
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
MAX_TOKENS = 4096
BATCH_SIZE = 5
SYSTEM_PROMPT = ""


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature=0.0):
    """Generates a message using the specified model."""
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
        }
    )
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body


def query_api(example, fpath):
    example['prompt'] = example['prompt']
    message = [{"role": "user", "content": example['prompt']}]
    response = generate_message(bedrock_runtime, MODEL_ID, SYSTEM_PROMPT, message, MAX_TOKENS)
    if response:
        output = response.get('content', [{}])[0].get('text', 'Error or no response')
        example['output'] = output
        print(f"Completed {fpath}")
        with open(fpath, 'w') as f:
            f.write(json.dumps(example, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lastfm-stage2-judge")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    print("Arguments:")
    print('=' * 50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('=' * 50)

    dataset_fn = os.path.join("output", args.task.split('_judge')[0], args.model_name.split("/")[-1], f"{args.split}.json",)
    print(f"Loading data from {dataset_fn}")
    dataset = load_data[args.task](dataset_fn)
    formatting_func = load_formatting_func[args.task]
    print("Example formatting:")
    print(formatting_func(dataset[0]))

    output_dir = os.path.join("output", args.task, args.model_name.split("/")[-1], args.split)
    print(f"Saving data to {output_dir}")
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

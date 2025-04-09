import argparse
import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.dataset import load_data, load_formatting_func

batch_size = 128
max_length = 4096
max_new_tokens = 8192


def setup_vllm(model_name, split):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize the LLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
        max_seq_len_to_capture=8192,
        dtype='half',
        trust_remote_code=True,
    )

    # Set the sampling parameters
    if split == 'test':
        # Set temperature to 0.0 for deterministic outputs
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    else:
        sampling_params = SamplingParams(temperature=0.9, max_tokens=max_new_tokens, top_p=0.95)

    return llm, tokenizer, sampling_params


def inference(task, model_name, split):

    data = load_data[task](split=args.split)
    formatting_func = load_formatting_func[args.task]
    dataset = []
    for i, item in enumerate(data):
        temp = item.copy()
        temp['prompt'] = formatting_func(temp)['prompt']
        dataset.append(temp)

    llm, tokenizer, sampling_params = setup_vllm(model_name, split)

    # Perform batch inference and write results
    output_file = os.path.join("output", args.task, args.model_name.split("/")[-1], f"{split}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if split == 'test':
        with open(output_file, 'w+') as f:
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:i+batch_size]
                batch_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": x['prompt']}],
                                                               tokenize=False, add_generation_prompt=True) for x in batch]
                outputs = llm.generate(batch_prompts, sampling_params)
                for j, o in enumerate(outputs):
                    example = batch[j]
                    example['output'] = o.outputs[0].text.strip()
                    f.write(json.dumps(example) + '\n')
                    f.flush()
    else:
        with open(output_file, 'w+') as f:
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:i + batch_size]
                batch_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": x['prompt']}],
                                                               tokenize=False, add_generation_prompt=True) for x in batch]
                outputs_1 = llm.generate(batch_prompts, sampling_params)
                outputs_2 = llm.generate(batch_prompts, sampling_params)
                for j, (o1, o2) in enumerate(zip(outputs_1, outputs_2)):
                    example = batch[j]
                    example["output_1"] = o1.outputs[0].text.strip()
                    example["output_2"] = o2.outputs[0].text.strip()
                    f.write(json.dumps(example) + "\n")
                    f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lastfm_stage2")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split", type=str, default='test')
    args = parser.parse_args()

    print("Arguments:")
    print('=' * 50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('=' * 50)

    inference(args.task, args.model_name, args.split)

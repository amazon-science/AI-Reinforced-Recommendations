"""Script for fine-tuning a causal language model using SFT."""

import os
import argparse
import torch
from accelerate import Accelerator
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.dataset import load_formatting_func, load_data


def train(args: argparse.Namespace) -> None:
    """Fine-tune a causal language model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Set environment variables and initialize accelerator
    os.environ["WANDB_PROJECT"] = args.task
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("Parsed Arguments:")
        print("=" * 50)
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print("=" * 50)

    # Load dataset
    train_dataset = load_data[args.task]("train")
    val_dataset = load_data[args.task]("val")

    # convert into huggingface dataset format
    formatting_func = load_formatting_func[args.task]
    train_dataset = Dataset.from_list([formatting_func(example) for example in train_dataset])
    val_dataset = Dataset.from_list([formatting_func(example) for example in val_dataset])

    if accelerator.is_main_process:
        print("Sample training data:", train_dataset[0])

    # Model setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device_map = {"": accelerator.process_index}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    # Parameter-efficient fine-tuning setup
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=[
            "k_proj", "v_proj", "q_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, peft_config)

    # Training configuration
    output_dir = os.path.join("output", args.task, args.model_name.split("/")[-1])
    training_args = SFTConfig(
        packing=True,
        max_seq_length=args.cutoff_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=args.eval_step,
        save_total_limit=10,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="wandb",
        run_name=args.model_name.split("/")[-1],
        gradient_checkpointing_kwargs={"use_reentrant": True},
        save_only_model=True,
        ddp_find_unused_parameters=False,
    )

    # Trainer initialization and training
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    trainer.train()

    torch.cuda.empty_cache()
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint_dir)

    # Push the fine-tuned model to the Hugging Face Hub
    if args.repo_id and accelerator.is_main_process:
        print("Pushing the model to the Hugging Face Hub...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model_to_merge = PeftModel.from_pretrained(
            base_model,
            final_checkpoint_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        merged_model = model_to_merge.merge_and_unload()
        merged_model.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)
    else:
        print("Not the main process. Skipping push to Hugging Face Hub.")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument("--task", type=str, default="lastfm-stage1")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--eval_step", type=float, default=0.05)
    parser.add_argument("--repo_id", type=str, default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

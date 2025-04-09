import argparse
import os
import torch
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
from datasets import Dataset
from src.dataset import load_data, load_formatting_func


def train(args):

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
    train_dataset = load_data[args.task]("train", args.num_iter)
    val_dataset = load_data[args.task]("val", args.num_iter)

    # convert into huggingface dataset format
    formatting_func = load_formatting_func[args.task]
    train_dataset = Dataset.from_list([formatting_func(example) for example in train_dataset])
    val_dataset = Dataset.from_list([formatting_func(example) for example in val_dataset])

    if accelerator.is_main_process:
        print("Sample training data:", train_dataset[0])

    # Model Setup
    model_name = f"xxhe/{args.task}-iter{args.num_iter-1}" if args.num_iter > 1 else f"xxhe/{args.task}-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": accelerator.process_index}
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    # Parameter Efficient Fine-Tuning
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)

    # Training Configuration
    output_dir = os.path.join("output", args.task, model_name.split("/")[-1])
    training_args = DPOConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_length=args.cutoff_len,
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
        report_to="wandb",
        run_name=model_name.split("/")[-1],
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
        beta=1.0,
    )

    # Trainer Initialization
    trainer = DPOTrainer(
        model=base_model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    torch.cuda.empty_cache()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Push the fine-tuned model to the Hugging Face Hub
    if args.repo_id and accelerator.is_main_process:
        print(f"Pushing the model {args.repo_id} to the Hugging Face Hub...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        model_to_merge = PeftModel.from_pretrained(base_model, output_dir, device_map="auto", torch_dtype=torch.float16,)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lastfm-stage2-dpo")
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--eval_step", type=float, default=0.05)
    parser.add_argument("--repo_id", type=str, default="")
    args = parser.parse_args()

    train(args)

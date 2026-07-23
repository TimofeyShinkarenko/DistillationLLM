import torch
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "unsloth/gemma-3-1b-it"
DATASET_NAME = "HuggingFaceTB/Countdown-Task-GOLD"


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float32,
        max_seq_length=4096,
        load_in_4bit = False,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 64,
        lora_dropout = 0,
        use_gradient_checkpointing = "unsloth", 
        random_state = 1234,
    )

    train_data = load_dataset(DATASET_NAME, "verified_Qwen2.5-7B-Instruct", split="train[:29000]")
    eval_data = load_dataset(DATASET_NAME, "verified_Qwen2.5-7B-Instruct", split="train[29000:]")

    def format_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False) 
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    train_data = train_data.map(format_chat_template, batched=True)
    eval_data = eval_data.map(format_chat_template, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        eval_dataset = eval_data,
        dataset_text_field = "text",
        dataset_num_proc = 4,
        args = TrainingArguments(
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs = 10,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            eval_strategy = "steps",
            eval_steps = 300,
            save_strategy = "steps",
            save_steps = 300,
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            optim = "adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            output_dir = "outputs_sft",
        ),
    )

    trainer.train()

    model.save_pretrained("gemma_sft_lora")
    tokenizer.save_pretrained("gemma_sft_lora")


if __name__ == "__main__":
    main()
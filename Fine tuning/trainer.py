import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from torch.optim.lr_scheduler import LambdaLR
import math
import random
import nlpaug.augmenter.word as naw
from datasets import load_metric
import numpy as np

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Prompt template
alpaca_prompt = """for the instruction generate a response from the input. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Data preparation function
def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Load and prepare dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Data Augmentation
def augment_data(example):
    aug = naw.SynonymAug(aug_src='wordnet')
    aug_text = aug.augment(example['instruction'])
    example['augmented_instruction'] = aug_text
    return example

augmented_dataset = dataset.map(augment_data, batched=False)
augmented_dataset = augmented_dataset.filter(lambda example: random.random() < 0.2)
combined_dataset = dataset.concatenate_datasets([dataset, augmented_dataset])
combined_dataset = combined_dataset.shuffle(seed=42)

# Cosine Annealing LR Scheduler with Warm Restarts
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

# Evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_metric = load_metric("rouge")
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    bertscore_metric = load_metric("bertscore")
    bertscore_results = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    
    return {
        "rouge1": rouge_results["rouge1"].mid.fmeasure,
        "rouge2": rouge_results["rouge2"].mid.fmeasure,
        "rougeL": rouge_results["rougeL"].mid.fmeasure,
        "bertscore_f1": np.mean(bertscore_results["f1"])
    }

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        seed=3407,
        output_dir="outputs",
        gradient_checkpointing=True,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

# Set up the learning rate scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer=trainer.optimizer,
    num_warmup_steps=100,
    num_training_steps=len(trainer.train_dataset) * trainer.args.num_train_epochs // trainer.args.gradient_accumulation_steps,
    num_cycles=trainer.args.num_train_epochs
)
trainer.lr_scheduler = scheduler

# Set up the compute_metrics function
trainer.compute_metrics = compute_metrics

# Train the model
trainer_stats = trainer.train()

# Print training stats
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

# Enable faster inference
FastLanguageModel.for_inference(model)

# Inference pipeline
def generate_response(instruction, input_text="", max_length=100):
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Example usage
print("\nSingle Inference Example:")
instruction = "Explain the concept of machine learning in simple terms."
response = generate_response(instruction)
print(f"Instruction: {instruction}")
print(f"Response: {response}")

# Batch inference example
print("\nBatch Inference Example:")
instructions = [
    "What is the capital of France?",
    "Explain the theory of relativity briefly.",
    "Give me a recipe for chocolate chip cookies."
]

responses = [generate_response(instr) for instr in instructions]
for instr, resp in zip(instructions, responses):
    print(f"Instruction: {instr}")
    print(f"Response: {resp}\n")

# Save the model
model.save_pretrained("gemma_finetuned_model")
tokenizer.save_pretrained("gemma_finetuned_model")

print("Fine-tuning complete and model saved!")
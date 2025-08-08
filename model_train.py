import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, merge_and_unload
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
DATA_FILE = "training_dataset.json"
OUTPUT_DIR = "./unit-ai-phi3-mini"
MAX_SEQ_LEN = 1024
NUM_EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_RATIO = 0.03
EVAL_RATIO = 0.1
SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 3

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        "lm_head", "embed_tokens"
    ],
)

model = get_peft_model(model, lora_config)

def format_instruction(sample):
    instr = sample.get("instruction") or sample.get("prompt") or ""
    out = sample.get("output") or sample.get("completion") or ""
    return f"<|user|>\n{instr}\n<|end|>\n<|assistant|>\n{out}\n<|end|>"

raw = load_dataset("json", data_files=DATA_FILE, split="train")
raw = raw.map(lambda x: {"text": format_instruction(x)})
ds = raw.train_test_split(test_size=EVAL_RATIO, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c != "input_ids" and c != "attention_mask" and c != "text"])
eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in eval_ds.column_names if c != "input_ids" and c != "attention_mask" and c != "text"])

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_32bit",
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=0.001,
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type="constant",
    logging_steps=LOGGING_STEPS,
    logging_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    dataloader_pin_memory=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

print("Merging LoRA weights into the base model...")
merged_model = merge_and_unload(trainer.model)
os.makedirs(OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved merged model to {OUTPUT_DIR}")

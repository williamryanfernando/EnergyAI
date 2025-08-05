import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

model_name = "microsoft/phi-3-mini-4k-instruct"
dataset_name = "training_dataset.json"  
new_model_name = "unit-ai-phi3-mini"  

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto", 
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

def format_instruction(sample):
	return f"<|user|>\n{sample['instruction']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"

dataset = load_dataset("json", data_files=dataset_name, split="train")

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,     
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    optim="paged_adamw_32bit",
    save_steps=50,         
    logging_steps=10,       
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,             
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", 
    max_seq_length=1024,      
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=format_instruction, 
)

print("Starting model fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

print(f"Saving fine-tuned model to ./{new_model_name}")
trainer.model.save_pretrained(new_model_name)
print("Model saved successfully.")
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
from peft import LoraConfig

#Load the prompt template content
with open("prompts/ip_op_prediction.txt", "r") as f:
    prompt_template_content = f.read()

model_id = "google/medgemma-27b-text-it"
max_seq_length = 5000

# Quantization config for 4-bit training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

#Load Model and Tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# Load Dataset
print("Loading dataset...")
dataset = load_from_disk("hf_dataset")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Preprocess
def format_dataset(examples):
    """
    Convert raw examples into formatted chat strings.
    Each example becomes: <user_prompt><assistant_response>
    """
    formatted_texts = []
    for notes, label in zip(examples["notes"], examples["class"]):
        # Insert notes into the prompt template
        user_content = prompt_template_content.replace("[collated_notes]", notes)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append(text)

    return {"text": formatted_texts}

print("Formatting datasets...")
train_dataset = train_dataset.map(
    format_dataset,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Formatting train dataset"
)
test_dataset = test_dataset.map(
    format_dataset,
    batched=True,
    remove_columns=test_dataset.column_names,
    desc="Formatting test dataset"
)

# Gemma's standard response template is "<start_of_turn>model\n"
response_template = "<start_of_turn>model\n"

# Data Collator for Assistant-Only Loss
# This manually masks all tokens except the assistant response with -100
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class DataCollatorForAssistantOnlyLM:
    """
    Data collator that masks labels for everything except assistant responses.
    Loss will only be computed on the assistant's reply (OP/IP classification).
    """
    tokenizer: Any
    response_template: str
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # SFTTrainer already tokenized the data, so features contain 'input_ids'
        # We just need to apply padding and create masked labels

        # Extract input_ids from features
        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]

        # Pad the sequences
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        labels = tokenized["input_ids"].clone()

        # Tokenize the response template to find where assistant response starts
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )

        for i in range(len(labels)):
            response_start_idx = None

            for idx in range(len(labels[i]) - len(response_token_ids) + 1):
                if labels[i][idx:idx + len(response_token_ids)].tolist() == response_token_ids:
                    response_start_idx = idx + len(response_token_ids)
                    break

            # Mask everything before the assistant response with -100
            if response_start_idx is not None:
                labels[i, :response_start_idx] = -100
            else:
                # If we can't find the response template, mask everything
                labels[i, :] = -100

            # Also mask padding tokens
            labels[i][labels[i] == self.tokenizer.pad_token_id] = -100

        tokenized["labels"] = labels
        return tokenized

# Instantiate the custom data collator
data_collator = DataCollatorForAssistantOnlyLM(
    tokenizer=tokenizer,
    response_template=response_template
)

# Training Arguments
training_args = SFTConfig(
    output_dir="./ip_op_model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,
    optim="paged_adamw_8bit",
    report_to="wandb",
    dataset_text_field="text",
    max_length=max_seq_length,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 2}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    peft_config=peft_config,  # PEFT config
    processing_class=tokenizer
)

print("Starting training...")
trainer.train()

# Save
print("Saving model...")
trainer.save_model("./ip_op_model_final")
tokenizer.save_pretrained("./ip_op_model_final")

print("Training complete!")

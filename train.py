from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import torch

# --- 1. Define Device (Standard practice for Apple Silicon) ---
DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

dataset = load_dataset("data", split='train')

def format_chat_template(batch, tokenizer):

    system_prompt =  """You are a helpful, honest and harmless assistant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }
    
base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        token="<your hugging face token>",
)

train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=5)

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map=DEVICE, # Use the defined MPS device
    dtype=DTYPE, # Load the model in bfloat16 to save memory
    token="<your hugging face token>",
    cache_dir="./workspace",
)


# print(next(model.parameters()).device)

model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 


training_args = TrainingArguments(
    output_dir="./Llama-3.2-1B-SFT-results", # Use a local folder name
    
    per_device_train_batch_size=2, # Start small: 1 or 2
    gradient_accumulation_steps=4, # Simulate a batch size of 8 (2*4)
    bf16=True,                     # CRITICAL: Enables bfloat16 for M4 GPU training
    fp16=False,                    # Must be False if bf16 is True
    num_train_epochs=5,             # Reduced to prevent overfitting (Start low)
    learning_rate=2e-4,             # Standard learning rate for QLoRA/LoRA
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",            # Use default AdamW optimizer
    report_to="none",
)


# 4. Initialize and Train
trainer = SFTTrainer(
    model, 
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B-SFT", num_train_epochs=50), 
    peft_config=peft_config,
)

trainer.train()  # Train the model

# 5. Save Final Model
trainer.save_model('complete_checkpoint') # Saves the adapter
trainer.model.save_pretrained("final_model") # Saves the adapter again (redundant, but safe)
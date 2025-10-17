import sys
import logging
import os
import datasets
from datasets import load_dataset
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from typing import Dict, List
import wandb
from datetime import datetime
import json


###################
# Hyper-parameters
###################
processed_dataset_path = "./MTP/Datasets/am-distilled-8192"
model_path = "./"
output_dir = "./ckpts"

# Conditional report_to based on process rank
is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
report_to = "wandb" if is_main_process else []

training_config = {
    "do_eval": False,
    "learning_rate": 1e-4,
    "log_level": "info",
    "logging_steps": 10,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 0.8,
    "max_steps": -1,
    "output_dir": output_dir,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 1000,
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 16,
    "warmup_ratio": 0.1,
    "dataloader_drop_last": True,
    "max_grad_norm": 1.0,
    "report_to": report_to,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
}
sft_config = SFTConfig(**training_config)


###################
# Wandb Configuration
###################
def init_wandb():
    """Initialize wandb for experiment tracking"""
    # Only initialize wandb on the main process (rank 0)
    if is_main_process:
        # Get current timestamp for run naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize wandb
        wandb.init(
            project="DMTD",  # Main project category
            name=f"{output_dir}_{timestamp}",  # Descriptive run name
        )

# Initialize wandb
init_wandb()

# Load model config and log to wandb (only on main process)
if is_main_process:
    model_config_path = os.path.join(model_path, "config.json")
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Log configurations by category
    # 1. Model configuration
    wandb.config.update({"model_config": model_config})
    
    # 2. Training configuration
    wandb.config.update({"train_config": training_config})
    
    # 3. Machine/Environment configuration
    machine_config = {
        "dataset_path": processed_dataset_path,
        "model_path": model_path,
        "output_dir": output_dir,
        "total_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    wandb.config.update({"machine_config": machine_config})

################
# Model Loading
################
checkpoint_path = model_path
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

# Freeze embedding and lm_head parameters
print("Freezing embedding and lm_head parameters...")
for param in model.model.embed_tokens.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

##################
# Data Loading
##################

dataset_dict = datasets.load_from_disk(processed_dataset_path)
train_dataset = dataset_dict['train']

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

# Start training
train_result = trainer.train()

# Log final metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Log additional final metrics to wandb (only on main process)
if is_main_process:
    wandb.log({
        "final_train_loss": metrics.get("train_loss", 0),
        "final_train_runtime": metrics.get("train_runtime", 0),
        "final_train_samples_per_second": metrics.get("train_samples_per_second", 0),
        "final_train_steps_per_second": metrics.get("train_steps_per_second", 0),
    })

print("Training completed successfully!")
print(f"Final metrics: {metrics}")

# Finish wandb run (only on main process)
if is_main_process:
    wandb.finish()

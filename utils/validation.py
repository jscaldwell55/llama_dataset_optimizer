# llama_dataset_optimizer/utils/validation.py

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import tempfile
import os

def train_lora_adapter(model_name: str, dataset: Dataset, output_dir: str):
    """
    A simplified function to train a LoRA adapter for validation purposes.
    """
    print(f"--- Training LoRA adapter on dataset of size {len(dataset)} ---")
    
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. Setup PEFT LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Preprocess dataset
    def preprocess_function(examples):
        # We only need the text for language modeling
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) 
                 for msgs in examples['messages']]
        return tokenizer(texts, truncation=True, max_length=1024, padding="max_length")

    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 4. Set up Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1, # A single epoch is enough for a quick validation
        logging_steps=10,
        save_steps=100, # Avoid saving checkpoints
        learning_rate=2e-4,
        fp16=False, # Use bf16 for Ampere
        bf16=True,
        lr_scheduler_type="cosine",
        report_to="none", # Disable reporting to wandb/etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        # We don't need an eval dataset for this quick check
    )
    
    # 5. Train
    train_result = trainer.train()
    
    # Return training loss as a proxy for performance
    final_loss = train_result.training_loss
    print(f"--- Finished training. Final training loss: {final_loss:.4f} ---")
    return final_loss


def validate_dataset_improvement(original_dataset: Dataset, optimized_dataset: Dataset, test_model: str):
    """
    Trains two LoRA adapters (baseline vs. optimized) and compares their final
    training loss. A lower loss on the optimized set suggests it's more learnable.
    """
    print("\n" + "="*50)
    print("🚀 Starting A/B Validation Framework")
    print(f"Test Model: {test_model}")
    print(f"Original Dataset Size: {len(original_dataset)}")
    print(f"Optimized Dataset Size: {len(optimized_dataset)}")
    print("="*50 + "\n")

    # Create temporary directories for adapters
    with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as optimized_dir:
        # --- Train on random baseline ---
        # Select a random subset from the original data of the same size as the optimized set
        baseline_sample_size = len(optimized_dataset)
        if baseline_sample_size > len(original_dataset):
             print("Warning: Optimized dataset is larger than original. Using full original dataset for baseline.")
             baseline_sample_size = len(original_dataset)

        baseline_dataset = original_dataset.shuffle(seed=42).select(range(baseline_sample_size))
        
        print("\n[Phase 1/2] Training baseline adapter on a random sample...")
        baseline_loss = train_lora_adapter(test_model, baseline_dataset, baseline_dir)
        
        # --- Train on optimized dataset ---
        print("\n[Phase 2/2] Training adapter on the optimized dataset...")
        optimized_loss = train_lora_adapter(test_model, optimized_dataset, optimized_dir)

    # --- Compare results ---
    print("\n" + "="*50)
    print("📊 A/B Validation Results")
    print("="*50)
    print(f"Baseline (Random Sample) Final Loss: {baseline_loss:.4f}")
    print(f"Optimized Dataset Final Loss:       {optimized_loss:.4f}")

    loss_reduction = baseline_loss - optimized_loss
    percentage_improvement = (loss_reduction / baseline_loss) * 100 if baseline_loss > 0 else 0

    print(f"\nLoss Reduction: {loss_reduction:.4f}")
    if loss_reduction > 0:
        print(f"✅ The optimized dataset resulted in a {percentage_improvement:.2f}% lower final training loss.")
        print("This suggests the model learned the patterns in the optimized data more efficiently.")
    else:
        print(f"❌ The optimized dataset resulted in a {abs(percentage_improvement):.2f}% higher final training loss.")
        print("This may indicate that the optimization process removed valuable samples. Consider adjusting weights.")
    print("="*50 + "\n")
    
    return {
        "baseline_loss": baseline_loss,
        "optimized_loss": optimized_loss,
        "loss_reduction_percentage": percentage_improvement
    }
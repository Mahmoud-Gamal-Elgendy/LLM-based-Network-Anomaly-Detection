"""
================================================================================
MODULE 3: MODEL FINE-TUNING
================================================================================

Purpose:
    This module fine-tunes Microsoft's phi-2 (2.7B parameters) on textified 
    network traffic data for anomaly detection.

What this script does:
    1. Loads phi-2 model with 4-bit quantization (saves GPU memory)
    2. Applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    3. Prepares custom dataset from JSON files
    4. Configures training with HuggingFace Trainer API
    5. Fine-tunes the model with proper hyperparameters
    6. Saves model checkpoints and final adapter weights

Why we need this:
    - Pre-trained phi-2 knows general language, not network security
    - Fine-tuning adapts it to understand network traffic patterns
    - LoRA enables training on consumer GPUs (only ~1% of parameters trained)
    - This creates a domain-specific LLM for 6G network security

Techniques Used:
    - QLoRA: Quantized LoRA (4-bit quantization + LoRA adapters)
    - Instruction Tuning: Teaching the model to follow classification tasks
    - Gradient Checkpointing: Reduces memory usage during training
    - Mixed Precision Training: Faster training with fp16/bf16

================================================================================
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# HuggingFace libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import numpy as np

# Progress tracking
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model loading and quantization."""
    
    # Model selection
    model_id: str = "microsoft/phi-2"
    
    # Quantization settings (4-bit for memory efficiency)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4 quantization
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    
    # Device settings
    device_map: str = "auto"  # Automatically distribute layers across GPUs
    trust_remote_code: bool = True  # Required for phi-2


@dataclass
class LoRAConfig_Custom:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    
    # LoRA hyperparameters
    r: int = 16  # Rank of the update matrices (higher = more capacity)
    lora_alpha: int = 32  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.05  # Dropout for regularization
    
    # Target modules (which layers to apply LoRA to)
    # For phi-2, we target query, key, value, and output projections
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",    # Query projection
        "k_proj",    # Key projection  
        "v_proj",    # Value projection
        "dense",     # Output dense layer
        "fc1",       # Feed-forward layer 1
        "fc2"        # Feed-forward layer 2
    ])
    
    # Task type
    task_type: str = "CAUSAL_LM"  # Causal language modeling
    
    # Inference mode
    inference_mode: bool = False  # Set to False for training
    
    # Bias handling
    bias: str = "none"  # Don't train bias parameters


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Output directories
    output_dir: str = "./Models/phi2-kdd-lora"
    logging_dir: str = "./Models/logs"
    
    # Training hyperparameters
    # 1 epoch is sufficient for LoRA fine-tuning — the model sees every sample once.
    # More epochs risk overfitting and are unnecessary for instruction tuning.
    num_train_epochs: int = 1  # 1 epoch ≈ 1,575 steps on T4 ≈ 2-3 hours
    per_device_train_batch_size: int = 4  # Batch size per GPU
    per_device_eval_batch_size: int = 8  # Larger eval batch (no gradients → less memory)
    gradient_accumulation_steps: int = 4  # Accumulate gradients (effective batch = 4*4=16)
    
    # Learning rate
    learning_rate: float = 2e-4  # LoRA typically uses higher LR than full fine-tuning
    warmup_steps: int = 50  # Proportional to 1 epoch (~3% of steps)
    
    # Optimization
    optim: str = "paged_adamw_8bit"  # Memory-efficient AdamW
    weight_decay: float = 0.01  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Mixed precision
    # CRITICAL: fp16=False because the model already computes in float16 via
    # bnb_4bit_compute_dtype. Adding AMP GradScaler on top causes overflow
    # (loss * scale_factor > float16 max), which makes GradScaler skip EVERY
    # optimizer step → loss=0, grad_norm=nan, lr=0.
    fp16: bool = False
    bf16: bool = False  # Use bfloat16 (better for newer GPUs like A100)
    
    # Logging and evaluation
    logging_steps: int = 25  # Log every 25 steps (less I/O overhead)
    eval_steps: int = 500  # Evaluate every 500 steps (eval is EXPENSIVE: 22k samples)
    save_steps: int = 250  # Save checkpoint every 250 steps (crash recovery)
    eval_strategy: str = "steps"  # Evaluate during training
    save_strategy: str = "steps"
    
    # Checkpointing — saves every 250 steps so you can RESUME after a crash
    save_total_limit: int = 2  # Keep only last 2 checkpoints (save disk space)
    load_best_model_at_end: bool = False  # Disabled: requires eval at every save step
    metric_for_best_model: str = "eval_loss"  # Metric to determine best model
    
    # Memory optimization
    # We enable gradient checkpointing MANUALLY in setup_lora() with
    # use_reentrant=False (required for QLoRA). Setting this to False
    # prevents the Trainer from re-enabling it with wrong/missing kwargs.
    gradient_checkpointing: bool = False
    
    # Other settings
    report_to: str = "tensorboard"  # Report metrics to TensorBoard
    seed: int = 42  # For reproducibility


# ============================================================================
# DATASET PREPARATION
# ============================================================================

class NetworkAnomalyDataset:
    """
    Custom dataset class for network anomaly detection.
    
    This class loads the textified JSON data and prepares it for training.
    """
    
    def __init__(self, json_path: str, tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            json_path (str): Path to textified JSON file
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"\n{'='*70}")
        print(f"Loading Dataset: {Path(json_path).name}")
        print(f"{'='*70}")
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ Loaded {len(self.data):,} records")
        
        # Get label distribution
        self._print_label_distribution()
    
    def _print_label_distribution(self):
        """Print distribution of attack categories."""
        distribution = {}
        for item in self.data:
            category = item['attack_category']
            distribution[category] = distribution.get(category, 0) + 1
        
        print(f"\nLabel Distribution:")
        for label, count in sorted(distribution.items()):
            percentage = (count / len(self.data)) * 100
            print(f"  - {label:10s}: {count:6,} ({percentage:5.2f}%)")
    
    def format_prompt(self, item: Dict) -> str:
        """
        Format a single item into a training prompt.
        
        For instruction-tuned models, we combine instruction, input, and output.
        
        Args:
            item (Dict): Single data item
        
        Returns:
            str: Formatted prompt
        """
        # Format: Instruction + Input + Output
        # This teaches the model the complete task structure
        
        prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Output:
{item['output']}"""
        
        return prompt
    
    def get_dataset(self) -> Dataset:
        """
        Convert to HuggingFace Dataset format.
        
        Returns:
            Dataset: HuggingFace Dataset object
        """
        print(f"\n{'='*70}")
        print("Converting to HuggingFace Dataset Format...")
        print(f"{'='*70}")
        
        # Format all prompts
        print("\nFormatting prompts...")
        formatted_texts = []
        for item in self.data:
            prompt = self.format_prompt(item)
            formatted_texts.append(prompt)
        
        # Verify sample
        print(f"\nSample prompt (first 200 chars):")
        print(f"{formatted_texts[0][:200]}...")
        
        # Tokenize all at once
        print("\nTokenizing dataset...")
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        
        # ================================================================
        # CRITICAL FIX: Create labels explicitly instead of relying on
        # DataCollatorForLanguageModeling. Labels = input_ids for real
        # tokens, -100 for padding (so loss ignores padding positions).
        # ================================================================
        print("Creating labels (masking padding with -100)...")
        labels = []
        for input_ids, attn_mask in zip(tokenized['input_ids'], tokenized['attention_mask']):
            label = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(input_ids, attn_mask)
            ]
            labels.append(label)
        
        # Convert to HuggingFace Dataset
        print("Creating dataset...")
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        print(f"✓ Dataset ready for training")
        print(f"  - Total samples: {len(dataset):,}")
        print(f"  - Features: {dataset.column_names}")
        
        # Verify a sample
        sample = dataset[0]
        non_padding = sum(sample['attention_mask'])
        non_ignored_labels = sum(1 for l in sample['labels'] if l != -100)
        print(f"\n  Sample verification:")
        print(f"    - Input IDs length: {len(sample['input_ids'])}")
        print(f"    - Attention mask length: {len(sample['attention_mask'])}")
        print(f"    - Non-padding tokens: {non_padding}")
        print(f"    - Non-ignored labels: {non_ignored_labels}")
        print(f"    - First 10 tokens: {sample['input_ids'][:10]}")
        print(f"    - First 10 labels: {sample['labels'][:10]}")
        
        if non_ignored_labels == 0:
            raise RuntimeError(
                "ALL labels are -100! The model cannot learn anything. "
                "Check that your text is being tokenized correctly."
            )
        
        return dataset


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def setup_model_and_tokenizer(config: ModelConfig):
    """
    Load and configure phi-2 model with quantization.
    
    Args:
        config (ModelConfig): Model configuration
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n{'='*70}")
    print("LOADING PHI-2 MODEL")
    print(f"{'='*70}")
    
    # ========================================================================
    # STEP 1: Configure 4-bit Quantization
    # ========================================================================
    print("\n[1/3] Configuring 4-bit Quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
    )
    
    print(f"✓ Quantization configured:")
    print(f"  - Type: {config.bnb_4bit_quant_type}")
    print(f"  - Compute dtype: {config.bnb_4bit_compute_dtype}")
    print(f"  - Double quantization: {config.bnb_4bit_use_double_quant}")
    
    # ========================================================================
    # STEP 2: Load Tokenizer
    # ========================================================================
    print("\n[2/3] Loading Tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code
    )
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding
    tokenizer.padding_side = "right"  # Pad on the right side
    
    print(f"✓ Tokenizer loaded")
    print(f"  - Vocabulary size: {len(tokenizer):,}")
    print(f"  - Pad token: {tokenizer.pad_token}")
    
    # ========================================================================
    # STEP 3: Load Model
    # ========================================================================
    print("\n[3/3] Loading phi-2 Model (2.7B parameters)...")
    print("  ⏳ This may take a few minutes...")
    
    # Load model configuration first and set pad_token_id
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code
    )
    model_config.pad_token_id = tokenizer.pad_token_id  # Set pad token ID
    model_config.use_cache = False  # Disable cache for training
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        config=model_config,
        quantization_config=bnb_config,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float16,
        attn_implementation="eager"  # Use eager attention (more stable)
    )
    
    print(f"✓ Model loaded successfully")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Memory footprint: ~{total_params * 4 / 1e9:.2f} GB (before quantization)")
    print(f"  - After 4-bit quantization: ~{total_params * 0.5 / 1e9:.2f} GB")
    
    return model, tokenizer


def setup_lora(model, lora_config: LoRAConfig_Custom):
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: Base model
        lora_config (LoRAConfig_Custom): LoRA configuration
    
    Returns:
        model: Model with LoRA adapters
    """
    print(f"\n{'='*70}")
    print("APPLYING LoRA (LOW-RANK ADAPTATION)")
    print(f"{'='*70}")
    
    # Prepare model for k-bit training
    # CRITICAL FIX: Disable gradient checkpointing here to avoid conflict
    # with TrainingArguments. We'll enable it in Trainer with use_reentrant=False.
    print("\n[1/3] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=False  # Let Trainer handle this with correct kwargs
    )
    
    # CRITICAL FIX: Enable input grad requirement for frozen quantized layers.
    # Without this, gradients cannot flow from the frozen base model into LoRA adapters.
    print("\n[2/3] Enabling input gradients for LoRA...")
    model.enable_input_require_grads()
    
    # Configure LoRA
    print("\n[3/3] Configuring LoRA adapters...")
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=lora_config.inference_mode,
        bias=lora_config.bias
    )
    
    print(f"✓ LoRA configuration:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - Target modules: {', '.join(lora_config.target_modules)}")
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params
    
    print(f"\n✓ LoRA adapters applied successfully")
    print(f"\nParameter Efficiency:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable: {trainable_percentage:.2f}%")
    print(f"  - Memory savings: {100 - trainable_percentage:.2f}%")
    
    # ====================================================================
    # CRITICAL FIX: Enable gradient checkpointing HERE with guaranteed
    # use_reentrant=False. We do NOT let the Trainer handle this because
    # some transformers/PEFT versions silently ignore the kwargs parameter.
    # ====================================================================
    print(f"\n[4/4] Enabling gradient checkpointing (use_reentrant=False)...")
    import functools
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print(f"  ✓ Enabled via gradient_checkpointing_kwargs API")
    except TypeError:
        # Older transformers version that doesn't support the kwargs parameter
        model.gradient_checkpointing_enable()
        print(f"  ⚠ Fallback: patching _gradient_checkpointing_func directly")
    
    # Belt-and-suspenders: directly patch the checkpointing function on
    # EVERY module that has it, ensuring use_reentrant=False is used.
    patched = 0
    for module in model.modules():
        if hasattr(module, '_gradient_checkpointing_func'):
            module._gradient_checkpointing_func = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=False
            )
            patched += 1
    print(f"  ✓ Patched _gradient_checkpointing_func on {patched} module(s)")
    
    return model


# ============================================================================
# DEBUG CALLBACK — Verifies training is actually updating weights
# ============================================================================

class EarlyDebugCallback(TrainerCallback):
    """
    Callback that prints detailed diagnostics for the first 5 training steps.
    
    If loss=0 or grad_norm=nan appears, it raises an error immediately so you
    don't waste hours on a broken training run.
    """
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step <= 5 and logs:
            loss = logs.get("loss", None)
            grad = logs.get("grad_norm", None)
            lr = logs.get("learning_rate", None)
            
            print(f"\n  [DEBUG Step {state.global_step}]  "
                  f"loss={loss}  grad_norm={grad}  lr={lr}")
            
            if loss is not None and (loss == 0 or str(loss) == "0"):
                print("  ⚠ WARNING: loss is 0 — the model is NOT learning!")
            if grad is not None and (str(grad) == "nan"):
                print("  ⚠ WARNING: grad_norm is nan — gradients are broken!")
            # lr=0 at step 1-2 is NORMAL during warmup (lr ramps from 0 → target).
            # Only warn if lr is still 0 after warmup should have started increasing.
            if lr is not None and lr == 0 and state.global_step > 5:
                print("  ⚠ WARNING: learning_rate is still 0 — optimizer may be stuck!")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, 
                training_config: TrainingConfig):
    """
    Fine-tune the model using HuggingFace Trainer API.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_config (TrainingConfig): Training configuration
    
    Returns:
        Trainer: Trained model
    """
    print(f"\n{'='*70}")
    print("FINE-TUNING PHI-2 ON NETWORK ANOMALY DETECTION")
    print(f"{'='*70}")
    
    # Create output directories
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(training_config.logging_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Training Arguments
    # ========================================================================
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        logging_steps=training_config.logging_steps,
        logging_first_step=True,  # Log first step to verify training starts
        logging_dir=training_config.logging_dir,
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        gradient_checkpointing=training_config.gradient_checkpointing,
        # gradient_checkpointing is handled manually in setup_lora()
        # with guaranteed use_reentrant=False.
        report_to=training_config.report_to,
        seed=training_config.seed,
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    print(f"\n{'='*70}")
    print("Training Configuration:")
    print(f"{'='*70}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  Batch size per device: {training_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    print(f"  FP16: {training_config.fp16}")
    print(f"  Optimizer: {training_config.optim}")
    
    # ========================================================================
    # Data Collator
    # ========================================================================
    # Using default_data_collator since labels are already created in the dataset.
    # This avoids issues with DataCollatorForLanguageModeling where pad_token=eos_token
    # can cause ALL tokens (including real content) to be masked as -100.
    data_collator = default_data_collator
    
    # Test data collator with a sample batch
    print(f"\nTesting data collator...")
    test_batch = data_collator([train_dataset[0], train_dataset[1]])
    print(f"  - Batch keys: {test_batch.keys()}")
    print(f"  - Input IDs shape: {test_batch['input_ids'].shape}")
    print(f"  - Labels shape: {test_batch['labels'].shape}")
    print(f"  - Has -100 in labels: {(-100 in test_batch['labels'].flatten())}")
    non_ignored = (test_batch['labels'] != -100).sum().item()
    total_labels = test_batch['labels'].numel()
    print(f"  - Non-ignored labels: {non_ignored}/{total_labels} ({100*non_ignored/total_labels:.1f}%)")
    print(f"  - Sample label values: {test_batch['labels'][0][:20].tolist()}")
    
    if non_ignored == 0:
        raise RuntimeError(
            "ALL labels are -100! The model has no targets to learn from. "
            "Check dataset creation and tokenization."
        )
    
    # ========================================================================
    # Initialize Trainer
    # ========================================================================
    print(f"\n{'='*70}")
    print("Initializing Trainer...")
    print(f"{'='*70}")
    
    # Ensure model is in training mode
    model.train()
    
    # Verify trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n✓ Trainable parameter groups: {len(trainable_params)}")
    if len(trainable_params) > 0:
        print(f"  Sample: {trainable_params[0][:50]}...")
    else:
        raise RuntimeError("No trainable parameters found! LoRA setup failed.")
    
    # Test forward pass to verify loss computation
    print(f"\nTesting forward pass...")
    import torch
    with torch.no_grad():
        test_batch = data_collator([train_dataset[0], train_dataset[1]])
        # Move to device
        test_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in test_batch.items()}
        outputs = model(**test_batch)
        test_loss = outputs.loss.item()
        print(f"  - Loss computed: {test_loss:.4f}")
        print(f"  - Loss is valid: {not torch.isnan(outputs.loss) and test_loss > 0}")
    
    if torch.isnan(outputs.loss) or test_loss == 0:
        raise RuntimeError(f"Forward pass test failed! Loss={test_loss}")
    
    # ========================================================================
    # Verify gradient flow (catches the use_reentrant bug)
    # ========================================================================
    print(f"\nTesting gradient flow with a single backward pass...")
    model.train()
    grad_test_batch = data_collator([train_dataset[0]])
    grad_test_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                       for k, v in grad_test_batch.items()}
    grad_test_output = model(**grad_test_batch)
    grad_test_output.loss.backward()
    
    # Check if any LoRA parameter got a gradient
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0 and not np.isnan(grad_norm):
                has_grad = True
                print(f"  ✓ Gradient verified: {name[:60]}... norm={grad_norm:.6f}")
                break
    
    if not has_grad:
        raise RuntimeError(
            "Gradient flow test FAILED! No LoRA parameter received a valid gradient. "
            "This means training would produce loss=0, grad_norm=nan. "
            "Check gradient_checkpointing_kwargs and enable_input_require_grads."
        )
    
    # Clear gradients from the test
    model.zero_grad()
    print(f"  ✓ Gradient flow confirmed - training will work correctly")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyDebugCallback()],  # Verify first steps aren't broken
    )
    
    print(f"✓ Trainer initialized")
    print(f"  - Training samples: {len(train_dataset):,}")
    print(f"  - Evaluation samples: {len(eval_dataset):,}")
    print(f"  - Total training steps: {len(train_dataset) // (training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps) * training_config.num_train_epochs:,}")
    
    # ========================================================================
    # Start Training
    # ========================================================================
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print("\nMonitor training progress:")
    print(f"  - TensorBoard: tensorboard --logdir {training_config.logging_dir}")
    print(f"  - Checkpoints will be saved to: {training_config.output_dir}")
    print(f"\n{'='*70}\n")
    
    # ── Resume from checkpoint if one exists (crash recovery) ──────────
    checkpoint_dir = Path(training_config.output_dir)
    last_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"\n  ✓ Resuming from checkpoint: {last_checkpoint}")
    
    # Train the model
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    return trainer


def save_model(trainer, tokenizer, output_dir: str):
    """
    Save the fine-tuned model and tokenizer.
    
    Args:
        trainer: Trained model
        tokenizer: Tokenizer to save
        output_dir (str): Directory to save model
    """
    print(f"\n{'='*70}")
    print("Saving Model...")
    print(f"{'='*70}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the model
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to: {output_dir}")
    print(f"✓ Tokenizer saved to: {output_dir}")
    
    # Print saved files
    saved_files = list(Path(output_dir).glob("*"))
    print(f"\nSaved files:")
    for file in saved_files:
        size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file.name} ({size:.2f} MB)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "PHI-2 FINE-TUNING PIPELINE")
    print(" "*10 + "Network Anomaly Detection (6G-AIOps)")
    print("="*70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Paths
    TRAIN_JSON = PROJECT_ROOT / "Data" / "train_textified_detailed_instruction.json"
    TEST_JSON = PROJECT_ROOT / "Data" / "test_textified_detailed_instruction.json"
    OUTPUT_DIR = PROJECT_ROOT / "Models" / "phi2-kdd-lora"
    
    # Configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig_Custom()
    training_config = TrainingConfig()
    
    # ========================================================================
    # STEP 1: Load Model and Tokenizer
    # ========================================================================
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # ========================================================================
    # STEP 2: Apply LoRA
    # ========================================================================
    model = setup_lora(model, lora_config)
    
    # ========================================================================
    # STEP 3: Prepare Datasets
    # ========================================================================
    print(f"\n{'='*70}")
    print("PREPARING DATASETS")
    print(f"{'='*70}")
    
    # ── max_length=256 ──────────────────────────────────────────────────
    # Your samples average ~117 tokens. Using 512 means 77% of every sequence
    # is useless padding that still costs GPU time. 256 fits >99% of samples
    # and cuts compute roughly in half.
    MAX_LENGTH = 256
    
    # Load training dataset
    train_data_loader = NetworkAnomalyDataset(
        str(TRAIN_JSON),
        tokenizer,
        max_length=MAX_LENGTH
    )
    train_dataset = train_data_loader.get_dataset()
    
    # Load evaluation dataset
    eval_data_loader = NetworkAnomalyDataset(
        str(TEST_JSON),
        tokenizer,
        max_length=MAX_LENGTH
    )
    eval_dataset_full = eval_data_loader.get_dataset()
    
    # ── Subsample eval set ──────────────────────────────────────────────
    # Evaluating on all 22,544 samples takes ~20 min per eval run.
    # Using 3,000 stratified samples gives a reliable loss estimate in ~3 min.
    EVAL_SUBSET_SIZE = 3000
    if len(eval_dataset_full) > EVAL_SUBSET_SIZE:
        eval_dataset = eval_dataset_full.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
        print(f"  ✓ Eval subset: {EVAL_SUBSET_SIZE:,} / {len(eval_dataset_full):,} samples")
    else:
        eval_dataset = eval_dataset_full
    
    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        training_config
    )
    
    # ========================================================================
    # STEP 5: Save Final Model
    # ========================================================================
    save_model(trainer, tokenizer, str(OUTPUT_DIR / "final"))
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print(f"\n{'='*70}")
    print(" "*20 + "ALL DONE!")
    print(f"{'='*70}")
    print(f"\nFine-tuned model ready for inference!")
    print(f"Model location: {OUTPUT_DIR / 'final'}")
    print(f"\nNext steps:")
    print(f"  1. Review training logs in: {training_config.logging_dir}")
    print(f"  2. Proceed to Module 4: Evaluation")
    print(f"  3. Test the model on new network traffic samples")
    print(f"\n{'='*70}\n")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the fine-tuning pipeline.
    
    Usage:
        python 3_train_model.py
    
    Requirements:
        - GPU with at least 8GB VRAM (for 4-bit quantization)
        - CUDA installed
        - All dependencies from requirements.txt
    
    Expected training time:
        - GPU (RTX 3080): ~2-3 hours for 3 epochs
        - GPU (A100): ~1 hour for 3 epochs
    """
    main()

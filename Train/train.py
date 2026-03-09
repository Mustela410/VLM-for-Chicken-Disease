import os
import torch
import random
import numpy as np
from functools import partial
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from config import *
from dataset import ChickenDiseaseDataset
from collator import data_collator
from callbacks import TrainLossCallback, CustomEvalCallback
from utils import find_latest_checkpoint


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256*28*28,
        max_pixels=MAX_PIXELS
    )
    print("Processor loaded")
    
    print(f"Loading model: {MODEL_ID}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("Model loaded")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Model VRAM: {vram_gb:.2f} GB")
    
    print("Configuring model...")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print("Gradient checkpointing enabled")
    
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = ChickenDiseaseDataset(TRAIN_JSONL, DATA_ROOT, processor)
    val_ds = ChickenDiseaseDataset(VAL_JSONL, DATA_ROOT, processor)
    
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=False,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
    )
    
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(data_collator, processor=processor),
        callbacks=[
            TrainLossCallback(),
            CustomEvalCallback(eval_at_step=EVAL_AT_STEP),
        ]
    )
    
    resume_from_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    
    print("Starting training...")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        print("Starting fresh training")
    
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Saving final model...")
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    
    print(f"Model saved to: {final_path}")
    print("Training complete")


if __name__ == "__main__":
    print("QWEN2-VL Training")
    main()

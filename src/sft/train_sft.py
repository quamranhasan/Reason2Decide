import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,  
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers.trainer_utils import set_seed

def load_json_dataset(train_file, valid_file):
    data_files = {"train": train_file, "validation": valid_file}
    return load_dataset("json", data_files=data_files)


def preprocess(example, tokenizer, max_input_length):
    input_text = example["question"]
    label_text = example["label"]

    # Format input
    predict_input = "predict: " + input_text

    # Tokenize input
    predict_input_enc = tokenizer(
        predict_input, padding=False, truncation=True, max_length=max_input_length
    )

    # Tokenize label
    label_enc = tokenizer(label_text, padding=False, truncation=True, max_length=256)

    return {
        "input_ids": predict_input_enc["input_ids"],
        "attention_mask": predict_input_enc["attention_mask"],
        "labels": label_enc["input_ids"],
    }


def create_data_collator(tokenizer):
    """
    Custom data collator for standard fine-tuning (no rationale, no multitask).
    Handles padding and -100 masking for labels.
    """
    def collator(features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad inputs
        batch = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt"
        )

        # Pad labels and replace pad token with -100
        label_batch = tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )
        labels_masked = label_batch["input_ids"].masked_fill(
            label_batch["input_ids"] == tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels_masked

        return batch

    return collator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    # print(f"seed:{args.seed}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    raw_datasets = load_json_dataset(args.train_file, args.valid_file)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Replace -100 with pad token
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        # Handle empty predictions
        decoded_preds = [p if p else "unknown" for p in decoded_preds]
        
        # Compute metrics
        acc = accuracy_score(decoded_labels, decoded_preds)
        f1 = f1_score(decoded_labels, decoded_preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": f1}
    # Debug: ensure 'question' is string
    sample_question = raw_datasets["train"][0]["question"]
    assert isinstance(sample_question, str), f"Expected string, got {type(sample_question)}: {sample_question}"

    # Tokenize
    tokenized = raw_datasets.map(
        lambda ex: preprocess(ex, tokenizer, args.max_input_length),
        batched=False
    )

    # Use custom collator
    collator = create_data_collator(tokenizer)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)


    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        max_steps=args.max_steps,
        save_steps=args.eval_steps,
        save_total_limit=5,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs/"),
        logging_steps=50,
        report_to="tensorboard",
        seed=args.seed,
        bf16=True, 
        predict_with_generate=True,
        gradient_accumulation_steps=args.grad_steps,
        
        remove_unused_columns=False  
    )

    # Monkey-patch torch.load to always allow full unpickling for this run
    # old_torch_load = torch.load
    # def torch_load_weights_only_false(*args, **kwargs):
    #     kwargs['weights_only'] = False
    #     return old_torch_load(*args, **kwargs)
    # torch.load = torch_load_weights_only_false

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator, 
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()


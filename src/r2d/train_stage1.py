
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import set_seed

def load_json_dataset(train_file, valid_file):
    data_files = {"train": train_file, "validation": valid_file}
    return load_dataset("json", data_files=data_files)

def preprocess_function(examples, tokenizer, max_input_length, max_output_length):
    inputs = [f"explain: {q}" for q in examples["question"]]
    targets = examples["rationale"]

    model_inputs = tokenizer(
        inputs,
        padding=False,
        truncation=True,
        max_length=max_input_length,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=False,
            truncation=True,
            max_length=max_output_length,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    raw_datasets = load_json_dataset(args.train_file, args.valid_file)

    # Preprocess: only rationale generation
    tokenized_datasets = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_input_length, args.max_output_length),
        batched=True,
        remove_columns=["question", "label", "rationale"],
        num_proc=4,
    )

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        max_steps=args.max_steps,
        save_total_limit=5,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs/"),
        logging_steps=50,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.grad_steps,
    )

    # Monkey-patch torch.load to always allow full unpickling for this run
    # old_torch_load = torch.load
    # def torch_load_weights_only_false(*args, **kwargs):
    #     kwargs['weights_only'] = False
    #     return old_torch_load(*args, **kwargs)
    # torch.load = torch_load_weights_only_false


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Stage 1 model saved to {args.output_dir}")

if __name__ == "__main__":
    main()



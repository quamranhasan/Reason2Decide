
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,  
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers.trainer_utils import set_seed
from transformers import TrainerCallback


# ---------------- Dataset Loading ----------------
def load_json_dataset(train_file, valid_file):
    data_files = {"train": train_file, "validation": valid_file}
    return load_dataset("json", data_files=data_files)

# ---------------- Preprocessing ----------------
def preprocess_function(examples, tokenizer, max_input_length):
    # Predict inputs (question only)
    predict_inputs = [f"predict: {q}" for q in examples["question"]]
    targets_labels = examples["label"]

    # Explanation inputs (question + gold label)
    explain_inputs = [
        f"given label: {l}, explain: {q}" for q, l in zip(examples["question"], examples["label"])
    ]
    targets_rationales = examples["rationale"]

    predict_enc = tokenizer(
        predict_inputs, padding=False, truncation=True, max_length=max_input_length
    )
    explain_enc = tokenizer(
        explain_inputs, padding=False, truncation=True, max_length=max_input_length
    )

    return {
        "predict_input_ids": predict_enc["input_ids"],
        "predict_attention_mask": predict_enc["attention_mask"],
        "predict_labels_text": targets_labels,
        "explain_input_ids": explain_enc["input_ids"],
        "explain_attention_mask": explain_enc["attention_mask"],
        "explain_labels_text": targets_rationales,
        "original_question": examples["question"], 
    }

# ---------------- Data Collator ----------------
def make_dss_data_collator(tokenizer, eval_mode=False):
    def collator(features):
        if eval_mode:
            # Eval: prediction only for eval loss
            batch = tokenizer.pad(
                [{"input_ids": f["predict_input_ids"],
                  "attention_mask": f["predict_attention_mask"]}
                 for f in features],
                return_tensors="pt"
            )
            labels = tokenizer(
                [f["predict_labels_text"] for f in features],
                padding=True, truncation=True, max_length=256,
                return_tensors="pt"
            )
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
            batch["labels"] = labels["input_ids"]
            return batch
        else:
            # Train: multitask (predict + explain)
            predict_batch = [{"input_ids": f["predict_input_ids"],
                              "attention_mask": f["predict_attention_mask"]} for f in features]
            explain_batch = [{"input_ids": f["explain_input_ids"],
                              "attention_mask": f["explain_attention_mask"]} for f in features]
            batch = {
                "predict": tokenizer.pad(predict_batch, return_tensors="pt"),
                "explain": tokenizer.pad(explain_batch, return_tensors="pt"),
                "predict_labels_text": [f["predict_labels_text"] for f in features],
                "explain_labels_text": [f["explain_labels_text"] for f in features],
                "questions": [f["original_question"] for f in features],   ### NEW
            }
            return batch
    return collator



class DelayedEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping but only after a delay (e.g., after warmup + mix steps).
    """
    def __init__(self, patience, warmup_steps, mix_steps, metric_name="eval_loss", greater_is_better=False):
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.mix_steps = mix_steps
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Only start checking after warmup + mix steps
        start_checking_step = self.warmup_steps + self.mix_steps
        if state.global_step < start_checking_step:
            return control  # skip early stopping during warmup/mix

        metric_val = metrics.get(self.metric_name)
        if metric_val is None:
            return control

        # Determine if we improved
        if self.best_score is None:
            self.best_score = metric_val
            self.counter = 0
        else:
            improved = (metric_val > self.best_score) if self.greater_is_better else (metric_val < self.best_score)
            if improved:
                self.best_score = metric_val
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered at step {state.global_step}")
            control.should_training_stop = True
        return control


# ---------------- Logging Callback ----------------
class RationaleLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        device = args.device
        import random
        samples = random.sample(range(len(self.eval_dataset)), self.num_samples)
        for idx in samples:
            ex = self.eval_dataset[idx]
            question = ex["question"]
            # Predict label
            pred_input = self.tokenizer(f"predict: {question}", return_tensors="pt").to(device)
            pred_ids = model.generate(**pred_input, max_length=50)
            pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            # Predict rationale conditioned on predicted label
            expl_prompt = f"given label: {pred_text}, explain: {question}"
            expl_input = self.tokenizer(expl_prompt, return_tensors="pt").to(device)
            expl_ids = model.generate(**expl_input, max_length=100)
            expl_text = self.tokenizer.decode(expl_ids[0], skip_special_tokens=True)
            print(f"\n[Sample {idx}]")
            print(f"Predicted Label: {pred_text}")
            print(f"Predicted Rationale: {expl_text}\n")

# ---------------- Custom Trainer ----------------
class DSS_Trainer(Seq2SeqTrainer):
    def __init__(self, alpha=0.5, alpha_warmup_steps=1000,
                 pred_label_transition_steps=3000,
                 train_collator=None, eval_collator=None, tokenizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_alpha = alpha
        self.alpha_warmup_steps = alpha_warmup_steps
        self.pred_label_transition_steps = pred_label_transition_steps
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        self.tokenizer = tokenizer


    def current_alpha(self):
        step = self.state.global_step
        if step >= self.alpha_warmup_steps:
            return self.target_alpha
        return self.target_alpha * step / max(1, self.alpha_warmup_steps)

    def get_train_dataloader(self):
        self.data_collator = self.train_collator
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        self.data_collator = self.eval_collator
        return super().get_eval_dataloader(eval_dataset)

    def fraction_predicted_labels(self):
        step = self.state.global_step
        if step < self.alpha_warmup_steps:
            return 0.0
        t = (step - self.alpha_warmup_steps) / max(1, self.pred_label_transition_steps)
        return min(0.9, t)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        device = self.args.device
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        # ========== Prediction Loss ==========
        pred_inputs = {k: v.to(device) for k, v in inputs["predict"].items()}
        pred_labels = self.tokenizer(
            inputs["predict_labels_text"], padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)
        pred_labels_ids = pred_labels["input_ids"]
        pred_labels_ids[pred_labels_ids == self.tokenizer.pad_token_id] = -100
        pred_out = model(**pred_inputs, labels=pred_labels_ids)

        # ========== Explanation Loss with scheduled sampling ==========
        frac_pred = self.fraction_predicted_labels()
        if frac_pred == 0.0:
            # Use gold explain inputs directly
            expl_inputs = {k: v.to(device) for k, v in inputs["explain"].items()}
        else:
            # Generate predicted labels
            core = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                gen_ids = core.generate(**pred_inputs, max_length=50, temperature=0.0)
            predicted_labels_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Mix gold and predicted per-example
            questions = inputs["questions"]
            expl_prompts = []
            for pl, gold, q in zip(predicted_labels_text, inputs["predict_labels_text"], questions):
                use_pred = (torch.rand(1).item() < frac_pred)  # choose predicted w.p. frac_pred
                label_for_prompt = pl if use_pred else gold
                expl_prompts.append(f"given label: {label_for_prompt}, explain: {q}")

            explain_enc = self.tokenizer(
                expl_prompts, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            expl_inputs = {"input_ids": explain_enc["input_ids"],
                           "attention_mask": explain_enc["attention_mask"]}
            

        expl_labels = self.tokenizer(
            inputs["explain_labels_text"], padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)
        expl_labels_ids = expl_labels["input_ids"]
        expl_labels_ids[expl_labels_ids == self.tokenizer.pad_token_id] = -100
        expl_out = model(**expl_inputs, labels=expl_labels_ids)



        alpha = self.current_alpha()

        loss = (1 - alpha) * expl_out.loss + alpha * pred_out.loss
        return (loss, {"pred_out": pred_out, "expl_out": expl_out}) if return_outputs else loss


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--stage1_model_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    set_seed(args.seed)
    print(args.seed)
    
    alpha_warmup_steps = int(0.05 * args.max_steps)
    pred_label_transition_steps = int(0.6 * args.max_steps)
    print(f"alpha warmup steps:{alpha_warmup_steps}, mix steps: {pred_label_transition_steps}")
    
    tokenizer = T5Tokenizer.from_pretrained(args.stage1_model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.stage1_model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")

    raw_datasets = load_json_dataset(args.train_file, args.valid_file)
    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_input_length),
        batched=True,
        remove_columns=["question", "label", "rationale"],
        num_proc=4,
    )

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

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        max_steps=args.max_steps,
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
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
        bf16=True,
        predict_with_generate=True,
        gradient_accumulation_steps=args.grad_steps,
    )

    train_collator = make_dss_data_collator(tokenizer, eval_mode=False)
    eval_collator = make_dss_data_collator(tokenizer, eval_mode=True)

    # Monkey-patch torch.load to always allow full unpickling for this run
    # old_torch_load = torch.load
    # def torch_load_weights_only_false(*args, **kwargs):
    #     kwargs['weights_only'] = False
    #     return old_torch_load(*args, **kwargs)
    # torch.load = torch_load_weights_only_false

    trainer = DSS_Trainer(
        alpha=args.alpha,
        alpha_warmup_steps=alpha_warmup_steps,
        model=model,
        tokenizer=tokenizer,   ### NEW
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        train_collator=train_collator,
        pred_label_transition_steps=pred_label_transition_steps,
        eval_collator=eval_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            DelayedEarlyStoppingCallback(
                patience=5, 
                warmup_steps=alpha_warmup_steps,
                mix_steps=pred_label_transition_steps,
                metric_name="eval_f1",  
                greater_is_better=True,   
            ),
            RationaleLoggingCallback(tokenizer, raw_datasets["validation"], num_samples=3)
        ]

    )

    trainer.train()

if __name__ == "__main__":
    main()


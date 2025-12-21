import pandas as pd
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def generate_outputs(model, tokenizer, inputs, device, batch_size, max_input_length, max_output_length):
    outputs_all = []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs[i:i + batch_size]
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                **encodings,
                max_length=max_output_length,
                temperature=0.0,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_all.extend(decoded)
    return outputs_all


def run_inference(args):
    # Device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load CSV
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} examples from {args.input_csv}")

    questions = df[args.question_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()

    # Prepare inputs
    model_inputs = [f"predict: {q}" for q in questions]

    print("Generating label predictions...")
    predicted_labels = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        inputs=model_inputs,
        device=device,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )

    y_true = [l.strip() for l in labels]
    y_pred = [p.strip() for p in predicted_labels]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy = {acc:.4f}")
    print(f"Macro F1 = {f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained or fine-tuned T5 model")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV file")
    parser.add_argument("--question_col", type=str, default="question",
                        help="Column name for input text")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Column name for ground-truth labels")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)


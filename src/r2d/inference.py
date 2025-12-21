import os
import torch
import torch.distributed as dist
import argparse
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


# ----------------------------
# Distributed helpers
# ----------------------------
def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if dist_is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def dist_init():
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

def barrier():
    if dist_is_initialized():
        dist.barrier()

def all_gather_list(obj_list_local):
    if not dist_is_initialized():
        return obj_list_local
    gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather, obj_list_local)
    all_items = []
    for part in gather:
        all_items.extend(part)
    return all_items


# ----------------------------
# Metrics
# ----------------------------
def compute_bertscore(predicted_rationales, gold_rationales, device):
    scorer = BERTScorer(lang="en", device=device)
    P, R, F1 = scorer.score(predicted_rationales, gold_rationales)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def compute_bleu_scores(predicted_rationales, gold_rationales):
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, gold in tqdm(
        zip(predicted_rationales, gold_rationales),
        total=len(predicted_rationales),
        desc="Computing BLEU"
    ):
        pred_tokens = nltk.word_tokenize(pred.lower())
        gold_tokens = nltk.word_tokenize(gold.lower())
        scores.append(sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothie))
    return sum(scores) / len(scores) if scores else 0.0


# ----------------------------
# Main
# ----------------------------
def main(args):
    dist_init()
    rank, world = get_dist_info()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device}, world_size={world}")

    # Ensure nltk tokenizer
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Load model
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load data
    df = pd.read_csv(args.input_csv)
    questions = df[args.question_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()
    rationales = df[args.rationale_col].astype(str).tolist()

    idx_all = list(range(len(questions)))
    idx_local = idx_all[rank::world]

    def take(lst, idxs):
        return [lst[i] for i in idxs]

    # =========================
    # 1. Generate labels
    # =========================
    predicted_labels_local = []
    with torch.no_grad():
        local_inputs = [f"predict: {q}" for q in take(questions, idx_local)]
        for i in tqdm(
            range(0, len(local_inputs), args.batch_size),
            desc="Generating Labels",
            disable=(rank != 0)
        ):
            batch = local_inputs[i:i + args.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=args.max_output_length_label)
            predicted_labels_local.extend(
                tokenizer.batch_decode(out, skip_special_tokens=True)
            )

    label_pairs = all_gather_list(list(zip(idx_local, predicted_labels_local)))

    if rank == 0:
        label_pairs.sort(key=lambda x: x[0])
        predicted_labels = [p for _, p in label_pairs]
        acc = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels, average="macro")
        print(f"\nLabel Accuracy = {acc:.4f}")
        print(f"Label Macro F1 = {f1:.4f}")

    barrier()

    # Broadcast labels
    if dist_is_initialized():
        obj = predicted_labels if rank == 0 else None
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        predicted_labels = obj_list[0]

    # =========================
    # 2. Generate rationales
    # =========================
    predicted_rationales_local = []
    with torch.no_grad():
        local_questions = take(questions, idx_local)
        local_labels = take(predicted_labels, idx_local)
        explain_inputs = [
            f"given label: {l}, explain: {q}"
            for q, l in zip(local_questions, local_labels)
        ]
        for i in tqdm(
            range(0, len(explain_inputs), args.batch_size),
            desc="Generating Rationales",
            disable=(rank != 0)
        ):
            batch = explain_inputs[i:i + args.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=args.max_output_length_rationale)
            predicted_rationales_local.extend(
                tokenizer.batch_decode(out, skip_special_tokens=True)
            )

    rat_pairs = all_gather_list(list(zip(idx_local, predicted_rationales_local)))

    if rank == 0:
        rat_pairs.sort(key=lambda x: x[0])
        predicted_rationales = [p for _, p in rat_pairs]

        print("\nEvaluating rationales...")
        bert_p, bert_r, bert_f1 = compute_bertscore(predicted_rationales, rationales, device)
        bleu = compute_bleu_scores(predicted_rationales, rationales)

        print(f"BERTScore P/R/F1: {bert_p:.4f} / {bert_r:.4f} / {bert_f1:.4f}")
        print(f"Average BLEU:     {bleu:.4f}")

        # Save outputs
        output_df = pd.DataFrame({
            args.question_col: questions,
            args.label_col: labels,
            args.rationale_col: rationales,
            "Predicted_Label": predicted_labels,
            "Predicted_Rationale": predicted_rationales
        })
        output_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")


# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed inference and rationale evaluation (no perplexity)"
    )

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--question_col", default="question")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--rationale_col", default="rationale")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length_label", type=int, default=50)
    parser.add_argument("--max_output_length_rationale", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


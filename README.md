# Reason2Decide
This repository contains the code and processed datasets accompanying our work on Reason2Decide.

## Repository Structure

```bash
.
├── processed_datasets.zip
│
├── src/
│   ├── sft/
│   │   ├── train_sft.py
│   │   └── inference.py
│   │
│   ├── r2d/
│   │   ├── train_stage1.py
│   │   ├── train_stage2.py
│   │   └── inference.py
│
├── requirements.txt
└── README.md
```

## Datasets
Included
* PubMedQA (processed)
* BioASQ Task B (processed)

The clinical triage dataset used in the paper is not publicly available and therefore cannot be redistributed. As a result, it is not included in this repository.

---

## Code Overview

### `src/sft/`

Contains code for standard supervised fine-tuning (SFT):

`train_sft.py`: SFT training

`inference.py`: SFT inference and evaluation


### `src/r2d/`

Contains code for Reason2Decide training and inference:

`train_stage1.py`: Rationale Generation Training

`train_stage2.py`: Joint Training for both rationale and prediction

`inference.py`: R2D inference and evaluation 

---

## Excluded Components

### DSS

The **distilling-step-by-step (DSS)** component is intentionally **not included** in this repository.

Reason:

* DSS in our experiments is a **direct modification of the original implementation**
* The original DSS source code is publicly available in the **authors’ official GitHub repository**

Please refer to the original authors’ repository for the DSS implementation. Can be found here: https://github.com/google-research/distilling-step-by-step

---

## Installation

We recommend using a Conda environment.
```bash
conda create -n reason2decide python=3.10
conda activate reason2decide
pip install -r requirements.txt
```

## Training and Inference

Training Script Usage:

For SFT and r2d_stage1 the following can be used:
```bash
torchrun --nproc_per_node=4 src/sft/train_sft.py \
    --train_file PATH_TO_TRAIN_FILE \
    --valid_file PATH_TO_VALID_FILE \
    --batch_size BATCH_SIZE \
    --grad_steps GRADIENT_ACCUMULATION_STEPS \
    --eval_steps EVAL_EVERY_N_STEPS \
    --max_steps MAX_TRAINING_STEPS \
    --model_name PRETRAINED_MODEL_NAME \
    --output_dir OUTPUT_DIR \
    --seed RANDOM_SEED \
    --max_input_length MAX_INPUT_TOKENS

```
For r2d_stage 2, the following can be used:
```bash
torchrun --nproc_per_node=NUM_GPUS src/r2d/train_stage2.py \
    --train_file PATH_TO_TRAIN_FILE \
    --valid_file PATH_TO_VALID_FILE \
    --batch_size BATCH_SIZE \
    --grad_steps GRADIENT_ACCUMULATION_STEPS \
    --eval_steps EVAL_EVERY_N_STEPS \
    --max_steps MAX_TRAINING_STEPS \
    --seed RANDOM_SEED \
    --stage1_model_dir PATH_TO_STAGE1_CHECKPOINT \
    --output_dir OUTPUT_DIR \
    --max_input_length MAX_INPUT_TOKENS
```

Inference Script Usage:

For SFT:
```bash
python src/sft/inference.py \
  --model_dir PATH_TO_MODEL \
  --input_file PATH_TO_INPUT_FILE \
  --batch_size BATCH_SIZE
```

For r2d:
```bash
torchrun --nproc_per_node=4 src/r2d/inference.py \
  --model_dir PATH_TO_MODEL \
  --input_csv PATH_TO_INPUT_FILE \
  --output_csv PATH_TO_OUTPUT_FILE
```

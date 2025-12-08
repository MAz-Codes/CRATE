# Running Training and Inference

This guide details how to train the Drum VAE model and generate new drum beats.

## Prerequisites
Ensure you have activated your virtual environment:
```bash
source .venv/bin/activate
```

## 1. Training

The training script `train.py` handles data downloading, preprocessing, and model training.

**Basic Command:**
```bash
python train.py
```

**Common Arguments:**
- `--epochs`: Number of training epochs (default: 10).
- `--batch_size`: Batch size (default: 16).
- `--seq_len`: Sequence length for tokenization (default: 512).
- `--file_limit`: Limit the number of files (useful for testing).
- `--output_dir`: Directory to save checkpoints (default: `checkpoints`).

**Example (Dry Run / Test):**
```bash
python train.py --epochs 1 --file_limit 10 --batch_size 2
```

## 2. Generation (Inference)

Use `generate.py` to create new drum MIDI files using a trained checkpoint.

**Basic Command:**
```bash
python generate.py --model_path checkpoints/final_model.pt
```

**Stylized Generation:**
You can condition the generation on a specific style:
```bash
python generate.py --model_path checkpoints/final_model.pt --style rock
python generate.py --model_path checkpoints/final_model.pt --style jazz --output_path generated_samples/jazz_beat.mid
```

**Tempo Control:**
Condition on a specific tempo (if supported by the trained model's vocabulary):
```bash
python generate.py --model_path checkpoints/final_model.pt --tempo 120
```

## 3. Evaluation

Use `evaluate.py` to calculate metrics (Loss, Perplexity, Accuracy) on the test set.

```bash
python evaluate.py --model_path checkpoints/final_model.pt
```
This will output overall metrics and per-style performance breakdown.
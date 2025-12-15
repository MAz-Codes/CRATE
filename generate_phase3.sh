#!/bin/bash
# Generate samples using the Phase 3 model

# Activate virtual environment if needed (assuming standard location)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

CHECKPOINT="checkpoints_phase3/best_model.pt"
OUTPUT_DIR="generated_samples_phase3"

mkdir -p $OUTPUT_DIR

echo "Generating samples from $CHECKPOINT..."

# Generate a few samples with different styles
# Styles: 0=rock, 1=funk, 2=jazz (approximate mapping, depends on dataset)

# Rock beat
python generate.py \
    --model_path $CHECKPOINT \
    --output_path "$OUTPUT_DIR/rock_beat.mid" \
    --style rock \
    --num_bars 8 \
    --temperature 0.8

# Funk beat
python generate.py \
    --model_path $CHECKPOINT \
    --output_path "$OUTPUT_DIR/funk_beat.mid" \
    --style funk \
    --num_bars 8 \
    --temperature 0.8

# Jazz beat
python generate.py \
    --model_path $CHECKPOINT \
    --output_path "$OUTPUT_DIR/jazz_beat.mid" \
    --style jazz \
    --num_bars 8 \
    --temperature 0.9

echo "Generation complete. Check $OUTPUT_DIR"

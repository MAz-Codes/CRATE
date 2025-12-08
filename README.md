# Drum VAE: Groove MIDI Drum Generator

This project implements a Variational Autoencoder (VAE) based on the Music Transformer architecture, specifically designed to generate realistic drum beats. It is trained on the **Groove MIDI Dataset (GMD)**, a large-scale dataset of human-performed drum grooves.

## Features
- **Drum-Focused Generation**: Specialized in generating MIDI drum patterns.
- **Style Conditioning**: Condition the generation on specific genres (e.g., Rock, Funk, Jazz, Hip-Hop).
- **Variational Autoencoder**: Utilizes a latent space to generate diverse and novel beats.
- **Groove MIDI Dataset**: Integration with TensorFlow Datasets to load high-quality drum performances.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd drumVAE
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a GPU-compatible version of PyTorch installed if you plan to train on GPU.*

## Usage

### Training
To train the model from scratch:

```bash
python train.py --epochs 100 --batch_size 16 --d_model 512
```
The dataset will be automatically downloaded to the `dataset/` folder on the first run.

### Generation
To generate a drum beat using a trained model:

```bash
python generate.py --model_path checkpoints/final_model.pt --output_path generated_samples/my_beat.mid --style funk
```
*   `--style`: Specify a genre (e.g., `rock`, `jazz`, `funk`, `hiphop`).
*   `--tempo`: (Optional) Target tempo in BPM.

### Evaluation
To evaluate the model's performance on the test set:

```bash
python evaluate.py --model_path checkpoints/final_model.pt
```

## Dataset
This project uses the **Groove MIDI Dataset** (`groove/full-midionly`), managed via TensorFlow Datasets. The data is stored locally in the `dataset/` directory.

## License
[License Information]

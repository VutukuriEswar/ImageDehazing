## Image Dehazing with U-Net

This project provides a powerful, local tool for removing haze from images using a deep learning model. It implements a U-Net architecture enhanced with Squeeze-and-Excitation blocks to achieve high-quality dehazing results, all runnable from your command line.

## Key Features

🧠 **Advanced Dehazing Model**
- Utilizes a U-Net architecture, a state-of-the-art model for image-to-image tasks.
- Incorporates Squeeze-and-Excitation (SE) blocks for improved channel-wise attention and feature representation.

📈 **Robust Training & Evaluation**
- Trains and evaluates models using standard metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
- Automatically saves the best-performing models based on PSNR, SSIM, and a combined score.
- Supports resuming training from checkpoints, so you never lose progress.

⚡ **Flexible & Fast Inference**
- Dehaze single images or entire directories of images with a simple command.
- Leverage your GPU for lightning-fast processing with CUDA and mixed-precision (AMP) support.
- Choose between different trained models (e.g., best PSNR vs. best SSIM) depending on your needs.

💻 **Local & Private**
- Run everything locally on your machine. No need to upload your images to a cloud service.
- Your data stays private and secure.

## Tech Stack

**Core Libraries:**
- **PyTorch & Torchvision**: For building the neural network and handling image data.
- **Pillow (PIL)**: For image manipulation and file I/O.
- **Scikit-image**: For calculating PSNR and SSIM metrics.
- **NumPy**: For efficient numerical operations.
- **Matplotlib**: For visualizing training progress and results.
- **TQDM**: For displaying sleek progress bars.

**Architecture:**
- **U-Net**: The primary neural network for image dehazing.
- **Squeeze-and-Excitation (SE) Block**: An attention mechanism to improve model performance.

## Quick Start Guide

### Prerequisites
- Python 3.8+
- A dehazing dataset (e.g., RESIDE, I-HAZE, O-HAZE)
- NVIDIA GPU with CUDA (highly recommended for training, optional for inference)

### Installation Steps

1.  **Clone the repository**
    ```bash
    git clone https://github.com/VutukuriEswar/ImageDehazing
    cd ImageDehazing
    ```

2.  **Set up a virtual environment**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your dataset**
    Create a `data` directory in the project root and organize your paired hazy/clear images as follows (In case you want to add data other than something already present in the repository):
    ```
    data/
    ├── train/
    │   ├── input/      # Hazy training images
    │   └── target/     # Clear training images
    └── test/
        ├── input/      # Hazy test images
        └── target/     # Clear test images
    ```
    *(Ensure filenames for hazy and clear pairs match, e.g., `001_hazy.png` and `001_clear.png`)*

5.  **Train the model**
    Start training with default settings:
    ```bash
    python src/train.py
    ```
    The best models will be saved in the `./output/models/` directory.

6.  **Run inference on your own images**
    Dehaze a single image:
    ```bash
    python src/inference.py --model_path ./output/models/best_model_combined.pth --input_path path/to/your/hazy_image.jpg --output_path path/to/save/dehazed_image.jpg
    ```

## Usage Guide

**Training (`src/train.py`)**

- **Start Training**: `python src/train.py`
- **Custom Parameters**: `python src/train.py --epochs 50 --batch_size 4 --lr 1e-3`
- **Resume Training**: `python src/train.py --resume`

**Inference (`src/inference.py`)**

- **Process a Single Image**:
  ```bash
  python src/inference.py --model_path ./output/models/best_model_combined.pth --input_path path/to/hazy.jpg --output_path path/to/dehazed.jpg
  ```
- **Process a Directory of Images**:
  ```bash
  python src/inference.py --model_path ./output/models/best_model_combined.pth --input_dir ./my_hazy_photos --output_dir ./my_dehazed_photos --use_gpu
  ```

## Configuration Details

**Dataset Setup:**
- The project requires a dataset of paired hazy and clear images.
- The `DehazeDataset` class automatically pairs images from the `input` and `target` directories based on alphabetical order of their filenames.

**Model Checkpoints:**
- During training, three different "best" models are saved in `./output/models/`:
  - `best_model_psnr.pth`: The model that achieved the highest PSNR on the test set.
  - `best_model_ssim.pth`: The model that achieved the highest SSIM.
  - `best_model_combined.pth`: The model with the best combined score (recommended for general use).
- Use any of these `.pth` files with the `inference.py` script.

## Acknowledgments

- Thanks to the creators of **PyTorch** for the incredible deep learning framework.
- This project is based on the **U-Net** architecture proposed by Ronneberger et al. and the **Squeeze-and-Excitation** block by Hu et al.
- Gratitude to the providers of the **RESIDE** dataset for making this research possible.
- Special thanks to you for your interest in this project

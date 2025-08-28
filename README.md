# Embroidery Image Generation with GANs

A deep learning project that uses Generative Adversarial Networks (GANs) to transform logo images into realistic embroidered versions. This project implements a U-Net generator with a multi-scale discriminator to create high-quality embroidery textures from input logos.

## 🎯 Project Overview

This project trains a conditional GAN to generate embroidered versions of logo images. The model learns to transform flat logo designs into realistic embroidery patterns with proper texture, stitching patterns, and visual characteristics typical of machine embroidery.

### Key Features

- **U-Net Generator**: Deep convolutional network with skip connections for high-quality image translation
- **Multi-Scale Discriminator**: Enhanced discriminator architecture for better texture discrimination
- **Advanced Loss Functions**: Combines adversarial, L1, perceptual, and SSIM losses for superior results
- **Data Augmentation**: Comprehensive augmentation including rotation, scaling, flipping, and color adjustments
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Checkpoint Management**: Automatic saving and loading of best models
- **Validation Monitoring**: Train/validation split with performance tracking

## 🏗️ Architecture

### Generator (U-Net)
- **Input**: 3-channel RGB logo images (256x256)
- **Output**: 3-channel RGB embroidered images (256x256)
- **Architecture**: 8-layer U-Net with skip connections
- **Features**: Batch normalization, dropout, and residual connections

### Discriminator (Multi-Scale PatchGAN)
- **Input**: Concatenated input and target/generated images (6 channels)
- **Architecture**: Two-scale discriminator (35x35 and 70x70 patches)
- **Purpose**: Distinguishes between real embroidered images and generated ones

### Loss Functions
1. **Adversarial Loss**: BCE loss for GAN training
2. **L1 Loss**: Pixel-wise reconstruction loss (λ=200)
3. **Perceptual Loss**: VGG16-based feature matching (λ=10)
4. **SSIM Loss**: Structural similarity preservation (λ=5)

## 📁 Dataset Structure

The project expects aligned image pairs in the following format:

```
MSEmb_DATASET/
└── embs_s_aligned/
    └── train/
        ├── 02469.png  # Side-by-side: target embroidery (left) | input logo (right)
        ├── 02468.png
        └── ...
```

Each training image should be a horizontally concatenated pair:
- **Left half**: Target embroidered image
- **Right half**: Input logo image

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for batch_size=16

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd embo_train_model
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- Pillow >= 8.3.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- scikit-learn
- opencv-python

## 🎮 Usage

### Training

1. **Prepare your dataset**
   - Place aligned image pairs in `MSEmb_DATASET/embs_s_aligned/train/`
   - Each image should contain target embroidery (left) and input logo (right)

2. **Start training**
```bash
python train_embroidery_improved.py
```

3. **Monitor progress**
   - Training logs will show loss values and progress
   - Sample images saved to `sample_images/` after each epoch
   - Model checkpoints saved to `checkpoints/`

### Configuration

Key training parameters (modify in `train_embroidery_improved.py`):

```python
batch_size = 16          # Adjust based on GPU memory
epochs = 300            # Maximum training epochs
lambda_L1 = 200         # L1 loss weight
lambda_perceptual = 10  # Perceptual loss weight  
lambda_ssim = 5         # SSIM loss weight
lr = 0.0001            # Learning rate
patience = 20          # Early stopping patience
```

### Resume Training

The script automatically detects and loads existing models:
- `checkpoints/best_generator.pth` - Best generator model
- `checkpoints/embroidery_improved_epoch_*.pth` - Latest checkpoint

### Inference

To generate embroidered images from logos:

```python
import torch
from models.test_model import UnetGenerator
from PIL import Image
import torchvision.transforms as transforms

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64).to(device)
generator.load_state_dict(torch.load('checkpoints/best_generator.pth'))
generator.eval()

# Prepare input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

input_image = Image.open('your_logo.png').convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Generate embroidered version
with torch.no_grad():
    output_tensor = generator(input_tensor)
    
# Convert back to image
output_np = output_tensor[0].cpu().numpy().transpose(1, 2, 0)
output_np = (output_np + 1) / 2  # Denormalize
output_image = Image.fromarray((output_np * 255).astype('uint8'))
output_image.save('embroidered_logo.png')
```

## 📊 Output Files

### Checkpoints
- `checkpoints/best_generator.pth` - Best generator model (L1 loss based)
- `checkpoints/embroidery_improved_best.pth` - Complete best checkpoint
- `checkpoints/embroidery_improved_epoch_*.pth` - Regular epoch checkpoints
- `checkpoints/embroidery_improved_final_*.pth` - Final epoch checkpoint

### Sample Images
- `sample_images/epoch_*_sample_*.png` - Generated samples after each epoch
- Shows input, target, and generated images side by side

## 🔧 Model Components

### Core Models
- `models/test_model.py` - U-Net generator architecture
- `models/patchGAN.py` - Multi-scale PatchGAN discriminator

### Utilities
- `utils/align.py` - Aligned dataset loader for side-by-side images
- `utils/data_loader.py` - General data loading utilities
- `utils/losses.py` - Advanced loss functions
- `utils/MultiFolderStitchSegDataset.py` - Multi-folder dataset handling

## 🎨 Data Augmentation

The training includes comprehensive data augmentation:
- **Geometric**: Random rotation (±15°), scaling (0.8-1.2x), horizontal flipping
- **Photometric**: Brightness (0.8-1.2x) and contrast (0.8-1.2x) adjustments
- **Probabilistic**: Each augmentation applied with 50% probability

## 📈 Training Features

### Learning Rate Scheduling
Aggressive learning rate reduction:
- Epochs 0-49: 100% learning rate
- Epochs 50-99: 50% learning rate  
- Epochs 100-149: 20% learning rate
- Epochs 150-199: 10% learning rate
- Epochs 200+: 5% learning rate

### Early Stopping
- Monitors L1 validation loss
- Stops training after 20 epochs without improvement
- Automatically saves best model

### Validation Split
- 80% training data
- 20% validation data
- Stratified split for consistent evaluation

## 🖥️ Hardware Requirements

### Minimum
- GPU: 6GB VRAM
- RAM: 16GB
- Storage: 10GB for datasets and checkpoints

### Recommended
- GPU: 12GB+ VRAM (RTX 3080/4070 or better)
- RAM: 32GB
- Storage: 50GB+ for large datasets

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` from 16 to 8 or 4
   - Reduce image size from 256 to 128

2. **Dataset Not Found**
   - Ensure dataset is in `MSEmb_DATASET/embs_s_aligned/train/`
   - Check image format (PNG files with side-by-side layout)

3. **Model Loading Errors**
   - Delete incompatible checkpoints in `checkpoints/`
   - Model will initialize with random weights

4. **Poor Results**
   - Ensure sufficient training data (100+ image pairs minimum)
   - Check data quality and alignment
   - Adjust loss function weights

## 📝 Notes

- Training typically takes 2-6 hours depending on dataset size and hardware
- Best results achieved with 500+ high-quality aligned image pairs
- The model automatically handles checkpoint resumption for interrupted training
- Sample images are generated after each epoch for visual progress monitoring

## 🤝 Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate documentation
3. Test with sample datasets
4. Update this README if needed

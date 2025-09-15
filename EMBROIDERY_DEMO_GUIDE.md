# 🎨 EMBROIDERY GENERATION SYSTEM - DEMO GUIDE

## 📋 SYSTEM OVERVIEW

Your embroidery generation system uses **CycleGAN** (Generative Adversarial Network) to convert regular images into realistic embroidery-style images.

---

## 🔄 COMPLETE WORKFLOW FLOWCHART

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBROIDERY GENERATION SYSTEM                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TRAINING      │    │   INFERENCE     │    │   POST-PROCESS  │
│   PHASE         │    │   PHASE         │    │   PHASE         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. DATA PREP    │    │ 1. LOAD MODEL   │    │ 1. CROP PADDING │
│                 │    │                 │    │                 │
│ • Load Dataset  │    │ • Load G_A      │    │ • Remove 3px    │
│ • Align Images  │    │ • Set to Eval   │    │   borders       │
│ • Normalize     │    │ • GPU/CPU       │    │ • Clean output  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 2. MODEL SETUP  │    │ 2. IMAGE INPUT  │    │ 2. SAVE RESULTS │
│                 │    │                 │    │                 │
│ • CycleGAN      │    │ • Load Image    │    │ • Save Original │
│ • Generator A   │    │ • Resize 256x256│    │ • Save Embroidery│
│ • Generator B   │    │ • Normalize     │    │ • 250x250 final │
│ • Discriminator │    │ • [-1,1] range  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ 3. TRAINING     │    │ 3. GENERATION   │
│                 │    │                 │
│ • Forward Pass  │    │ • G_A(input)    │
│ • GAN Loss      │    │ • Generate      │
│ • Cycle Loss    │    │ • Embroidery    │
│ • Backward Pass │    │ • Style Transfer│
│ • Update Weights│    │                 │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ 4. SAVE MODEL   │
│                 │
│ • Save G_A      │
│ • Save G_B      │
│ • Save D_A      │
│ • Save D_B      │
└─────────────────┘
```

---

## 🚀 DEMO STEPS

### **PHASE 1: TRAINING (One-time setup)**

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run training
python train_direct.py
```

**What happens during training:**
- Loads aligned embroidery dataset from `./MSEmb_DATASET/embs_s_aligned/train`
- Creates CycleGAN model with ResNet-9blocks generator
- Trains for 150 epochs + 150 decay epochs
- Saves model checkpoints in `./checkpoints/`

### **PHASE 2: INFERENCE (Generate embroidery)**

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run inference on all test images
python test_embroidery_only.py
```

**What happens during inference:**
- Loads trained Generator A model
- Processes all images in `./test_image/` folder
- Generates embroidery versions
- Saves results in `./embroidery_results/`

---

## 📁 FILE STRUCTURE

```
embo_train_model/
├── 📂 checkpoints/           # Trained models
│   ├── latest_net_G_A.pth    # Generator A (Normal → Embroidery)
│   ├── latest_net_G_B.pth    # Generator B (Embroidery → Normal)
│   ├── latest_net_D_A.pth    # Discriminator A
│   └── latest_net_D_B.pth    # Discriminator B
├── 📂 test_image/            # Input images for testing
│   ├── 00138.png
│   ├── 00146.png
│   └── ... (142 images)
├── 📂 embroidery_results/    # Generated embroidery images
│   ├── 00138_original.png
│   ├── 00138_embroidery.png
│   └── ... (284 files total)
├── 📂 MSEmb_DATASET/         # Training dataset
│   └── embs_s_aligned/train/
├── 🐍 train_direct.py        # Training script
├── 🐍 test_embroidery_only.py # Inference script
└── 📂 models/                # Model definitions
    ├── cycle_gan_model.py
    ├── base_model.py
    └── networks.py
```

---

## 🎯 KEY TECHNICAL DETAILS

### **Model Architecture:**
- **Generator A**: ResNet-9blocks (Normal → Embroidery)
- **Generator B**: ResNet-9blocks (Embroidery → Normal)  
- **Discriminator**: PatchGAN (70x70 patches)
- **Input/Output**: 256x256 RGB images

### **Training Process:**
1. **Forward Pass**: G_A(A) → B, G_B(B) → A
2. **Cycle Consistency**: G_B(G_A(A)) ≈ A
3. **Adversarial Loss**: D_A vs G_A, D_B vs G_B
4. **Identity Loss**: G_A(B) ≈ B (optional)

### **Image Processing:**
1. **Input**: Load image → Resize to 256x256 → Normalize to [-1,1]
2. **Generation**: Pass through Generator A
3. **Output**: Denormalize to [0,1] → Crop padding → Save as PNG

---

## 🎨 DEMO SCENARIOS

### **Scenario 1: Single Image Demo**
```bash
# Place your image in test_image/ folder
# Run inference
python test_embroidery_only.py
# Check results in embroidery_results/
```

### **Scenario 2: Batch Processing**
```bash
# Add multiple images to test_image/
# Run inference (processes all images)
python test_embroidery_only.py
# Get 284 output files (142 originals + 142 embroideries)
```

### **Scenario 3: Custom Training**
```bash
# Modify train_direct.py parameters
# Change epochs, learning rate, etc.
python train_direct.py
```

---

## 📊 EXPECTED RESULTS

### **Input Images:**
- Any RGB image (PNG, JPG, etc.)
- Automatically resized to 256x256
- Normalized to [-1, 1] range

### **Output Images:**
- Embroidery-style versions
- 250x250 pixels (cropped from 256x256)
- Same filename with `_embroidery` suffix
- Realistic thread-like texture and appearance

### **Performance:**
- **Training**: ~2-4 hours (depending on GPU)
- **Inference**: ~1-2 seconds per image
- **Memory**: ~4-6GB GPU memory during training

---

## 🔧 TROUBLESHOOTING

### **Common Issues:**
1. **CUDA out of memory**: Reduce batch size in train_direct.py
2. **Model not found**: Ensure training completed successfully
3. **White borders**: Fixed by cropping 3px padding
4. **Poor quality**: Train for more epochs or adjust learning rate

### **File Requirements:**
- ✅ Trained model checkpoints in `./checkpoints/`
- ✅ Test images in `./test_image/`
- ✅ Virtual environment activated
- ✅ PyTorch and dependencies installed

---

## 🎉 DEMO SUCCESS CRITERIA

✅ **Training completes** without errors  
✅ **Model checkpoints saved** in checkpoints/  
✅ **Inference runs** on test images  
✅ **Embroidery images generated** with realistic texture  
✅ **No white borders** in output images  
✅ **Results saved** in embroidery_results/  

---

## 📈 NEXT STEPS

1. **Improve Quality**: Train longer, adjust hyperparameters
2. **Add More Data**: Include more diverse embroidery styles
3. **Real-time Demo**: Create web interface for live generation
4. **Batch Processing**: Add command-line arguments for customization
5. **Quality Metrics**: Add SSIM, PSNR evaluation

---

*This system demonstrates successful image-to-image translation using CycleGAN for embroidery generation!* 🎨✨

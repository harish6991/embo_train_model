import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from models.unetModel import U2NETP
import cv2

class FullDatasetMaskGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # Load trained model
        self.model = U2NETP(3, 1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from: {model_path}")
            print(f"📊 Training epoch: {checkpoint.get('epoch', 'Unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded: {model_path}")

        self.model.eval()

        # Transform
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def generate_mask(self, image_path):
        """Generate mask for single image"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Transform and predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(input_tensor)
            main_pred = predictions[0][0, 0].cpu().numpy()

        # Resize to original size
        mask_resized = cv2.resize(main_pred, original_size, interpolation=cv2.INTER_LINEAR)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        return mask_uint8

    def process_dataset(self, input_folder, output_folder):
        """Generate masks for entire dataset"""
        os.makedirs(output_folder, exist_ok=True)

        # Get all images
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])

        image_files.sort()
        print(f"🔍 Found {len(image_files)} images to process")

        successful = 0
        for filename in tqdm(image_files, desc="Generating masks"):
            try:
                input_path = os.path.join(input_folder, filename)
                mask = self.generate_mask(input_path)

                # Save mask
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.png")

                Image.fromarray(mask, mode='L').save(output_path)
                successful += 1

            except Exception as e:
                print(f"❌ Error: {filename} - {e}")

        print(f"✅ Generated {successful} masks in {output_folder}")

def main():
    print("🎭 FULL DATASET MASK GENERATION")
    print("="*50)

    # Configuration
    model_path = "./best_u2net_mask_model.pth"

    generator = FullDatasetMaskGenerator(model_path)

    # Generate masks for different dataset splits
    datasets_to_process = [
        ("./MSEmb_DATASET/embs_s_unaligned/train/trainX_e", "./MSEmb_DATASET/embs_s_unaligned/train/generated_masks")
    ]

    for input_folder, output_folder in datasets_to_process:
        if os.path.exists(input_folder):
            print(f"\n📁 Processing: {input_folder}")
            generator.process_dataset(input_folder, output_folder)
        else:
            print(f"⚠️ Folder not found: {input_folder}")

    print("\n🎉 MASK GENERATION COMPLETED!")

if __name__ == "__main__":
    main()

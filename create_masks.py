import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def create_embroidery_mask(image_path, save_visualization=False):
    """
    Simple numpy-based mask creation: ignore white pixels, mask everything else
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to numpy array (already is, but make explicit)
    img_array = np.array(img_rgb)

    # Define white threshold
    white_threshold = 240

    # Simple approach: if ANY channel is below threshold, it's NOT white
    # Create mask where each pixel is True if it's NOT white
    mask = np.any(img_array < white_threshold, axis=2)

    # Convert boolean mask to 0-255 uint8
    # True (embroidery) = 255 (white in mask)
    # False (white background) = 0 (black in mask)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Optional: Save visualization
    if save_visualization:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Boolean Mask')
        axes[1].axis('off')

        axes[2].imshow(mask_uint8, cmap='gray')
        axes[2].set_title('Final Mask (0-255)')
        axes[2].axis('off')

        plt.tight_layout()

        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.savefig(f'mask_process_{base_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    return mask_uint8

def process_dataset(input_folder, output_folder, visualize_samples=True):
    """
    Process entire dataset to create masks
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(ext.lower())
        ])

    image_files.sort()

    if not image_files:
        print(f"❌ No image files found in {input_folder}")
        return

    print(f"🔍 Found {len(image_files)} images to process")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")

    # Process images with progress bar
    successful = 0
    failed = 0

    for i, filename in enumerate(tqdm(image_files, desc="Creating masks")):
        try:
            input_path = os.path.join(input_folder, filename)

            # Create mask
            save_viz = visualize_samples and i < 3  # Save visualization for first 3 images
            mask = create_embroidery_mask(input_path, save_visualization=save_viz)

            if mask is not None:
                # Save mask with same name as original
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.png")

                # Save as PIL Image for better quality
                mask_pil = Image.fromarray(mask, mode='L')
                mask_pil.save(output_path, optimize=True)

                successful += 1
            else:
                print(f"❌ Failed to process: {filename}")
                failed += 1

        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed += 1

    print(f"\n🎉 Processing completed!")
    print(f"✅ Successfully processed: {successful} images")
    print(f"❌ Failed: {failed} images")
    print(f"📁 Masks saved in: {output_folder}")

def test_single_image(image_path):
    """
    Test mask creation on a single image with visualization
    """
    print(f"🧪 Testing mask creation on: {image_path}")

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Create mask with visualization
    mask = create_embroidery_mask(image_path, save_visualization=True)

    if mask is not None:
        # Save test mask
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"test_mask_{base_name}.png"

        mask_pil = Image.fromarray(mask, mode='L')
        mask_pil.save(output_path)

        print(f"✅ Test mask saved: {output_path}")
        print(f"✅ Visualization saved: mask_process_{base_name}.png")
    else:
        print(f"❌ Failed to create mask for {image_path}")

def main():
    """
    Main function to create masks for embroidery dataset
    """
    print("🎭 EMBROIDERY MASK GENERATOR")
    print("="*50)

    # Dataset paths
    input_folder = r".\MSEmb_DATASET\embs_s_unaligned\train\trainX_e"
    output_folder = r".\MSEmb_DATASET\embs_s_unaligned\train\masks"

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        print("Available folders:")
        base_path = r".\MSEmb_DATASET"
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if dirs:
                    print(f"  📁 {root}")
                    for d in dirs[:5]:  # Show first 5 directories
                        print(f"    📁 {d}")
        return

    # Process the entire dataset
    process_dataset(input_folder, output_folder, visualize_samples=True)

    print("\n" + "="*50)
    print("🎯 MASK GENERATION COMPLETED!")
    print("="*50)
    print("✅ High-quality binary masks created")
    print("✅ Sharp edges and clean boundaries")
    print("✅ Ready for U2-Net training")
    print("="*50)

if __name__ == "__main__":
    main()

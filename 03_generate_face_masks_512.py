"""
==============================================================================
Suspect-Face Composite Generator: 512x512 Native Semantic Face Parsing Pipeline
Improvement 1: Generate High-Precision 512x512 Masks from Restored Faces
==============================================================================
Processes 2,000 512x512 restored faces from Indian_Face_512.zip.
Generates:
  1. Raw 19-class masks (512x512, uint8, classes 0-18)
  2. Part masks for human-in-the-loop inpainting (Nose, Eyes/Brows, Lips/Mouth, Hair, Skin)
  3. Visual segmentation overlays and comparison chart
  4. Final archive: Indian_Face_Masks_512.zip
==============================================================================
"""

import os
import sys
import time
import zipfile
import shutil
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import validated BiSeNet architecture from bisenet_model
from bisenet_model import BiSeNet, PART_GROUPS, COLOR_PALETTE

# ------------------------------------------------------------------------------
# 1. Configuration & Paths
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ZIP = os.path.join(BASE_DIR, "Dataset_01_Faces_512.zip") if os.path.exists(os.path.join(BASE_DIR, "Dataset_01_Faces_512.zip")) else os.path.join(BASE_DIR, "Indian_Face_512.zip")
MODEL_PATH = os.path.join(BASE_DIR, "parsing_bisenet.pth")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "Indian_Face_Masks_512")
OUTPUT_ZIP = os.path.join(BASE_DIR, "Dataset_04_Masks_512.zip")
PREVIEW_FILE = os.path.join(BASE_DIR, "previews", "03_face_parsing_512_preview.png")


RAW_MASKS_DIR = os.path.join(OUTPUT_FOLDER, "raw_masks")
PART_MASKS_DIR = os.path.join(OUTPUT_FOLDER, "part_masks")
VIS_DIR = os.path.join(OUTPUT_FOLDER, "visualizations")


# ------------------------------------------------------------------------------
# 2. Main Processing Routine
# ------------------------------------------------------------------------------
def run_512_segmentation():
    print("=" * 75)
    print("   SUSPECT-FACE COMPOSITE PIPELINE: 512x512 NATIVE FACE PARSING")
    print("=" * 75)
    
    # Multi-threading optimization for CPU
    torch.set_num_threads(min(12, os.cpu_count() or 8))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Compute Device: {device} (PyTorch {torch.__version__}, Threads: {torch.get_num_threads()})")
    
    if not os.path.exists(INPUT_ZIP):
        raise FileNotFoundError(f"Missing input archive: {INPUT_ZIP}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing BiSeNet weights at: {MODEL_PATH}")

    # Prepare clean output directories
    os.makedirs(RAW_MASKS_DIR, exist_ok=True)
    os.makedirs(PART_MASKS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load Model
    print(f"\n[1/4] Loading verified BiSeNet model ({os.path.getsize(MODEL_PATH)/(1024*1024):.1f} MB)...")
    net = BiSeNet(n_classes=19).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    net.eval()
    print("[SUCCESS] BiSeNet loaded and ready.")

    # Image Transform
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Read from zip archive directly
    print(f"\n[2/4] Reading images from {os.path.basename(INPUT_ZIP)}...")
    with zipfile.ZipFile(INPUT_ZIP, 'r') as z_in:
        names = sorted([f for f in z_in.namelist() if f.lower().endswith(('.png', '.jpg'))])
        print(f"      Total 512x512 faces found: {len(names)}")
        
        batch_size = 16
        t_start = time.time()
        preview_samples = []
        
        print(f"      Processing in batches of {batch_size}...")
        for i in tqdm(range(0, len(names), batch_size), desc="Generating 512x512 Masks"):
            batch_names = names[i:i + batch_size]
            tensors = []
            orig_rgbs = []
            valid_bases = []
            
            for fname in batch_names:
                buf = z_in.read(fname)
                img_bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                tensors.append(trans(Image.fromarray(img_rgb)))
                orig_rgbs.append(img_rgb)
                base = os.path.splitext(os.path.basename(fname))[0]
                valid_bases.append(base)
                
            if not tensors:
                continue
                
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            
            with torch.no_grad():
                out = net(batch_tensor)
                preds = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
                
            for b in range(len(valid_bases)):
                base = valid_bases[b]
                mask = preds[b]  # shape: (512, 512), values: 0-18
                
                # 1. Save Raw 19-class Indexed Mask
                cv2.imwrite(os.path.join(RAW_MASKS_DIR, f"{base}_mask.png"), mask)
                
                # 2. Save Aggregated Inpainting Part Masks (Binary 0 or 255)
                for part_name, class_ids in PART_GROUPS.items():
                    part_mask = np.isin(mask, class_ids).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(PART_MASKS_DIR, f"{base}_{part_name}.png"), part_mask)
                    
                # 3. Save Sample Visualizations (first 50)
                if i < 50 and b == 0:
                    color_mask = COLOR_PALETTE[mask]
                    overlay = cv2.addWeighted(orig_rgbs[b], 0.5, color_mask, 0.5, 0)
                    cv2.imwrite(os.path.join(VIS_DIR, f"{base}_vis.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
                if len(preview_samples) < 3 and b == 0:
                    preview_samples.append((orig_rgbs[b], COLOR_PALETTE[mask], mask))

    elapsed = time.time() - t_start
    print(f"\n[SUCCESS] Generated masks for {len(names)} faces in {elapsed:.1f}s ({len(names)/elapsed:.1f} faces/sec)!")

    # --------------------------------------------------------------------------
    # 3. Generate Visual Verification Preview
    # --------------------------------------------------------------------------
    print(f"\n[3/4] Creating verification preview: {os.path.basename(PREVIEW_FILE)}...")
    fig, axes = plt.subplots(len(preview_samples), 5, figsize=(18, 3.8 * len(preview_samples)))
    col_titles = [
        '512x512 Restored Face',
        '19-Class Semantic Overlay',
        'Nose Mask (Part)',
        'Eyes & Brows Mask (Part)',
        'Mouth & Lips Mask (Part)'
    ]

    for row, (orig, color_mask, mask) in enumerate(preview_samples):
        nose_mask = np.isin(mask, PART_GROUPS['nose']).astype(np.uint8) * 255
        eyes_mask = np.isin(mask, PART_GROUPS['eyes_brows']).astype(np.uint8) * 255
        mouth_mask = np.isin(mask, PART_GROUPS['mouth_lips']).astype(np.uint8) * 255

        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(color_mask)
        axes[row, 2].imshow(nose_mask, cmap='gray')
        axes[row, 3].imshow(eyes_mask, cmap='gray')
        axes[row, 4].imshow(mouth_mask, cmap='gray')

        for col in range(5):
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11, fontweight='bold', pad=10)

    plt.suptitle("Suspect-Face Composite: Native 512x512 Semantic Face Parsing Masks (Problem 3)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(PREVIEW_FILE, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Preview saved to: {PREVIEW_FILE}")

    # --------------------------------------------------------------------------
    # 4. Package ZIP Archive
    # --------------------------------------------------------------------------
    print(f"\n[4/4] Archiving masks into {os.path.basename(OUTPUT_ZIP)}...")
    shutil.make_archive(
        base_name=os.path.splitext(OUTPUT_ZIP)[0],
        format='zip',
        root_dir=OUTPUT_FOLDER
    )
    zip_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"[SUCCESS] Saved ZIP archive: {OUTPUT_ZIP} ({zip_size:.2f} MB)")
    print("=" * 75)
    print("  [COMPLETE] IMPROVEMENT 1 (512x512 FACE MASKS) FINISHED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_512_segmentation()

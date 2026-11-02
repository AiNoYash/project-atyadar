"""
==============================================================================
Suspect-Face Composite Generator: Semantic Face Parsing Pipeline
Problem 3: Generate Face Parsing Masks for Human-in-the-Loop Refinement
==============================================================================
Runs on Local PC (CPU) or GPU. Processes 2,000 face images and generates:
  1. Raw 19-class semantic masks (for fine-tuning Inpainting models)
  2. Aggregated part masks (Nose, Eyes/Brows, Lips/Mouth, Hair, Lower Face)
  3. Color-coded visual segmentation overlays
  4. ZIP archive: Indian_Face_Masks.zip
"""

import os
import sys
import glob
import time
import zipfile
import shutil
import urllib.request
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Configuration & Paths
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_CANDIDATES = glob.glob(os.path.join(BASE_DIR, "*Indian_Face*.zip"))
INPUT_FOLDER = os.path.join(BASE_DIR, "Indian_Face")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "Indian_Face_Masks")
OUTPUT_ZIP = os.path.join(BASE_DIR, "Indian_Face_Masks.zip")
PREVIEW_FILE = os.path.join(BASE_DIR, "face_parsing_preview.png")
MODEL_PATH = os.path.join(BASE_DIR, "parsing_bisenet.pth")

WEIGHTS_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth"

# Semantic Classes in CelebAMask-HQ / BiSeNet
# 0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g (glasses)
# 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck, 
# 15: neck_l, 16: cloth, 17: hair, 18: hat
PART_GROUPS = {
    "eyes_brows": [2, 3, 4, 5, 6],   # Eyes, eyebrows, glasses
    "nose": [10],                     # Nose
    "mouth_lips": [11, 12, 13],       # Upper lip, lower lip, mouth cavity
    "hair": [17, 18],                 # Hair and hats
    "face_skin": [1],                 # Cheeks, forehead, chin
}

# Distinct vivid color palette for 19 classes
COLOR_PALETTE = np.array([
    [0, 0, 0],        # 0: background (black)
    [204, 0, 0],      # 1: skin (red)
    [76, 153, 0],     # 2: l_brow (green)
    [0, 255, 0],      # 3: r_brow (lime)
    [51, 51, 255],    # 4: l_eye (blue)
    [204, 0, 204],    # 5: r_eye (magenta)
    [0, 255, 255],    # 6: glasses (cyan)
    [255, 204, 204],  # 7: l_ear (pink)
    [102, 51, 0],     # 8: r_ear (brown)
    [255, 0, 0],      # 9: earring
    [102, 204, 0],    # 10: nose (bright green)
    [255, 255, 0],    # 11: mouth (yellow)
    [0, 0, 153],      # 12: u_lip (navy)
    [0, 0, 204],      # 13: l_lip (blue)
    [255, 153, 51],   # 14: neck (orange)
    [0, 51, 0],       # 15: necklace
    [0, 204, 204],    # 16: cloth (teal)
    [153, 0, 153],    # 17: hair (purple)
    [255, 204, 153]   # 18: hat
], dtype=np.uint8)


# ------------------------------------------------------------------------------
# 2. BiSeNet Model Architecture
# ------------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = resnet18(weights=None)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        H0, W0 = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        feat4 = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(feat4)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = nn.functional.interpolate(avg, (feat32.size(2), feat32.size(3)), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = nn.functional.interpolate(feat32_sum, (feat16.size(2), feat16.size(3)), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = nn.functional.interpolate(feat16_sum, (H0 // 8, W0 // 8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
        return torch.mul(feat, atten) + feat


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetHead(256, 256, n_classes)
        self.conv_out16 = BiSeNetHead(128, 64, n_classes)
        self.conv_out32 = BiSeNetHead(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp16, feat_cp32 = self.cp(x)
        feat_out = self.ffm(feat_res8, feat_cp16)
        feat_out = self.conv_out(feat_out)
        feat_out = nn.functional.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        return feat_out


# ------------------------------------------------------------------------------
# 3. Model Loader & Auto-Downloader
# ------------------------------------------------------------------------------
def get_face_parser(device):
    """Downloads weights if missing, loads BiSeNet model into evaluation mode."""
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Pretrained weights not found. Downloading BiSeNet weights (~53 MB)...")
        print(f"       URL: {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, MODEL_PATH)
        print(f"[SUCCESS] Downloaded to {MODEL_PATH}")

    net = BiSeNet(n_classes=19)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()
    return net


# ------------------------------------------------------------------------------
# 4. Image Loading & Preprocessing
# ------------------------------------------------------------------------------
def load_face_images():
    """Loads images from directory or ZIP archive."""
    items = []
    if os.path.exists(INPUT_FOLDER) and len(os.listdir(INPUT_FOLDER)) > 0:
        print(f"[1/5] Loading face photos from folder: {INPUT_FOLDER}")
        for f in sorted(os.listdir(INPUT_FOLDER)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append((f, os.path.join(INPUT_FOLDER, f), False))
    elif ZIP_CANDIDATES:
        zip_path = ZIP_CANDIDATES[0]
        print(f"[1/5] Loading face photos from ZIP: {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if info.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not info.is_dir():
                    fname = os.path.basename(info.filename)
                    if fname:
                        raw = z.read(info.filename)
                        items.append((fname, raw, True))
    else:
        raise FileNotFoundError(f"No Indian_Face folder or ZIP found in: {BASE_DIR}")

    print(f"      Loaded {len(items)} source face images successfully.")
    return items


# ------------------------------------------------------------------------------
# 5. Core Segmentation & Mask Generation
# ------------------------------------------------------------------------------
def run_segmentation():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Initializing Face Parsing Model on device: {device.type.upper()}")
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count() or 12)
        print(f"       Optimized for {torch.get_num_threads()} CPU threads.")

    net = get_face_parser(device)
    items = load_face_images()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Subfolders for clean organization
    raw_masks_dir = os.path.join(OUTPUT_FOLDER, "raw_masks_19class")
    part_masks_dir = os.path.join(OUTPUT_FOLDER, "part_masks_human_in_loop")
    vis_dir = os.path.join(OUTPUT_FOLDER, "visualizations")
    os.makedirs(raw_masks_dir, exist_ok=True)
    os.makedirs(part_masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Visual Preview items
    preview_samples = []

    print(f"\n[2/5] Running semantic face segmentation on {len(items)} faces...")
    batch_size = 16 if device.type == 'cuda' else 8

    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size), desc="Segmenting Faces"):
            batch_items = items[i:i + batch_size]
            batch_tensors = []
            orig_imgs = []
            base_names = []

            for fname, source, is_bytes in batch_items:
                if is_bytes:
                    img_np = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img_np = cv2.imread(source)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                orig_imgs.append(img_rgb)
                base_names.append(os.path.splitext(fname)[0])

                # BiSeNet inference standard is 512x512
                pil_img = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
                tensor = transform(pil_img)
                batch_tensors.append(tensor)

            inputs = torch.stack(batch_tensors).to(device)
            outputs = net(inputs)
            preds = outputs.squeeze(0).cpu().numpy().argmax(1)  # shape: (B, 512, 512)

            for b_idx in range(len(batch_items)):
                base = base_names[b_idx]
                orig_h, orig_w = orig_imgs[b_idx].shape[:2]
                mask_512 = preds[b_idx].astype(np.uint8)

                # Resize mask back to original resolution (e.g. 224x224) using Nearest Neighbor
                mask_orig = cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # 1. Save Raw 19-class Indexed Mask (Values 0-18)
                cv2.imwrite(os.path.join(raw_masks_dir, f"{base}_mask.png"), mask_orig)

                # 2. Generate Aggregated Human-in-the-Loop Part Masks (Binary 0 or 255)
                # Useful directly in the UI when witness selects "Nose" or "Mouth"
                for part_name, class_ids in PART_GROUPS.items():
                    part_mask = np.isin(mask_orig, class_ids).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(part_masks_dir, f"{base}_{part_name}.png"), part_mask)

                # 3. Create Colorized Segmentation Overlay
                color_mask = COLOR_PALETTE[mask_orig]
                # Blend with original photo for visual inspection
                overlay = cv2.addWeighted(orig_imgs[b_idx], 0.5, color_mask, 0.5, 0)
                cv2.imwrite(os.path.join(vis_dir, f"{base}_vis.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                if len(preview_samples) < 3:
                    preview_samples.append((orig_imgs[b_idx], color_mask, mask_orig))

    t_inf = time.time() - t_start
    print(f"[SUCCESS] Segmented {len(items)} faces in {t_inf:.2f}s ({len(items)/t_inf:.1f} faces/sec)!")

    # --------------------------------------------------------------------------
    # 6. Generate Visual Preview Plot
    # --------------------------------------------------------------------------
    print(f"\n[3/5] Saving visual verification preview: {os.path.basename(PREVIEW_FILE)}...")
    fig, axes = plt.subplots(len(preview_samples), 4, figsize=(15, 3.8 * len(preview_samples)))
    col_titles = [
        'Original Photo',
        'Semantic Segmentation Overlay',
        'Nose Mask (Human-in-Loop)',
        'Eyes & Brows Mask'
    ]

    for row, (orig, color_mask, mask_orig) in enumerate(preview_samples):
        nose_mask = np.isin(mask_orig, PART_GROUPS['nose']).astype(np.uint8) * 255
        eyes_mask = np.isin(mask_orig, PART_GROUPS['eyes_brows']).astype(np.uint8) * 255

        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(color_mask)
        axes[row, 2].imshow(nose_mask, cmap='gray')
        axes[row, 3].imshow(eyes_mask, cmap='gray')

        for col in range(4):
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(PREVIEW_FILE, dpi=120)
    plt.close()
    print(f"[SUCCESS] Preview saved to: {PREVIEW_FILE}")

    # --------------------------------------------------------------------------
    # 7. Package ZIP Archive
    # --------------------------------------------------------------------------
    print(f"\n[4/5] Packaging masks archive to: {os.path.basename(OUTPUT_ZIP)}...")
    shutil.make_archive(
        base_name=os.path.splitext(OUTPUT_ZIP)[0],
        format='zip',
        root_dir=OUTPUT_FOLDER
    )
    print(f"[SUCCESS] Saved ZIP archive: {OUTPUT_ZIP} ({os.path.getsize(OUTPUT_ZIP)/(1024*1024):.2f} MB)")

    print("\n" + "="*70)
    print(" [5/5] PROBLEM 3 DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f" Source images processed: {len(items)}")
    print(f" Masks directory:         {OUTPUT_FOLDER}")
    print(f" Output ZIP archive:      {OUTPUT_ZIP}")
    print(f" Preview image:           {PREVIEW_FILE}")
    print(f" Total elapsed time:      {time.time()-t_start:.2f} seconds")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_segmentation()

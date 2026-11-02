# 🧬 Indian Suspect-Face Composite Generator
### Multi-Modal AI Forensic Pipeline: Sketch + Text ➔ Photorealistic Face with Human-in-the-Loop Refinement

---

## 📌 Executive Summary & Project Goal
Law enforcement and forensic investigators routinely rely on eyewitness descriptions to produce suspect composite sketches. However, conventional composite software (like FACES or Identi-Kit) relies on rigid, pre-packaged facial feature templates that struggle to capture authentic Indian ethnic characteristics, natural skin undertones, lighting variations, and organic hand-drawn sketches.

This project implements an end-to-end generative AI system that takes:
1. **Rough Hand-Drawn Sketches** (varying from clean forensic line-art to shaky, faded pencil drawings)
2. **Detailed Forensic Text Descriptions** (specifying gender, age, face shape, skin undertone, eye shape, nose bridge, facial hair, etc.)
3. **Optional Low-Resolution CCTV / Mobile Photos** (for enhancement and restoration)
4. **Interactive Human-in-the-Loop Region Refinement** (allowing an investigator or witness to select parts like the nose or eyes and refine them iteratively using semantic inpainting)

---

## 🏗️ Phase 1: What We Have Accomplished (Data Engineering & Multi-Modal Dataset)

All dataset engineering and preprocessing milestones are **100% complete, audited, and verified**.

### 1. Multi-Modal Dataset Inventory

| Modality / Problem | Standardized File | Count & Resolution | Description & Forensic Purpose |
| :--- | :--- | :--- | :--- |
| **Problem 1: Target Faces** | [`Dataset_01_Faces_512.zip`](Dataset_01_Faces_512.zip) | **2,000 photos**<br>(512×512, PNG) | **Super-Resolved & Denoised Ground Truth:** Restored from low-res 224px compressed JPEGs using CodeFormer (`w=0.7`). Eliminates compression noise while preserving genuine Indian ethnic identity, dermal textures, eye glints, and teeth margins. |
| **Problem 2: Forensic Sketches** | [`Dataset_03_Sketches_Augmented.zip`](Dataset_03_Sketches_Augmented.zip) | **6,000 sketches**<br>(3 styles / identity) | **Robust Structural Conditioning:** 2,000 original clean line-art + 2,000 hand-wobble shaky scribbles + 2,000 faded pencil dropouts mimicking witness memory lapses. |
| **Problem 3: Part Masks** | [`Dataset_04_Masks_512.zip`](Dataset_04_Masks_512.zip) | **12,000 masks**<br>(512×512, PNG) | **Interactive Inpainting Masks:** Native 512×512 semantic masks generated via BiSeNet. Includes 2,000 19-class raw masks + 10,000 isolated binary masks for `nose`, `eyes_brows`, `mouth_lips`, `hair`, and `face_skin`. |
| **Forensic Captions** | [`Dataset_02_Captions.zip`](Dataset_02_Captions.zip) | **2,000 captions**<br>(Text prompts) | **Semantic Conditioning:** Highly descriptive forensic prompts detailing skin tones (*"wheatish brown"*, *"dusky"*), jawlines, eye shapes, and facial hair. |
| **Source Raw Backups** | `Raw_Faces_224_Source.zip`<br>`Raw_Sketches_224_Source.zip` | 2,000 each (224×224) | Untouched original low-resolution inputs preserved for archival integrity. |

---

### 2. Multi-Modal Alignment & Partitioning Strategy

All 4 modalities are indexed in a single machine-readable manifest:
* **Manifest File:** [`dataset_manifest.jsonl`](dataset_manifest.jsonl) (6,000 paired lines)
* **Dataset Summary:** [`dataset_summary.json`](dataset_summary.json)

```json
{
  "id": "indian_0001",
  "split": "train",
  "style": "rough_wobble",
  "target_image": "faces_512/indian_0001.png",
  "conditioning_sketch": "sketches_augmented/indian_0001_aug1_rough.jpg",
  "prompt": "A man with light skin, a square face shape, a straight jawline, dark eyebrows, almond-shaped eyes, a medium-sized nose with a slight curve at the tip, thin lips, and clean-shaven facial hair.",
  "masks": {
    "raw_19class": "masks_512/raw_masks/indian_0001_mask.png",
    "nose": "masks_512/part_masks/indian_0001_nose.png",
    "eyes_brows": "masks_512/part_masks/indian_0001_eyes_brows.png",
    "mouth_lips": "masks_512/part_masks/indian_0001_mouth_lips.png",
    "hair": "masks_512/part_masks/indian_0001_hair.png",
    "face_skin": "masks_512/part_masks/indian_0001_face_skin.png"
  }
}
```

#### Strict Anti-Leakage Identity Splits:
* **Train Set (80%):** 1,600 unique identities ➔ **4,800 paired training samples**
* **Validation Set (10%):** 200 unique identities ➔ **600 evaluation samples**
* **Test Set (10%):** 200 unique identities ➔ **600 benchmark evaluation samples**
> *All 3 sketch variations for any given person are strictly locked to the same partition, preventing test-set data leakage.*

---

### 3. Visual Verification Artifacts

Visual verification charts are preserved in [`previews/`](previews/):
1. **[`previews/01_face_restoration_preview.png`](previews/01_face_restoration_preview.png):** Compares original 224px blurry photo vs 512px CodeFormer output with zoomed eye/mouth details.
2. **[`previews/02_sketch_augmentation_preview.png`](previews/02_sketch_augmentation_preview.png):** Visualizes the 3 forensic sketch augmentation styles.
3. **[`previews/03_face_parsing_512_preview.png`](previews/03_face_parsing_512_preview.png):** Demonstrates pixel-perfect 512×512 segmentation overlays and extracted part masks (`nose`, `eyes_brows`, `mouth_lips`).

---

## 🔬 Phase 2: Technical Roadmap (Model Selection & Fine-Tuning Strategy)

### 1. Hardware Context: Dual NVIDIA RTX 3090 (2 × 24GB = 48GB VRAM)
Your dual RTX 3090 rig provides 48 GB of high-speed GDDR6X VRAM and over 70 TFLOPS of FP16 compute. This hardware allows:
* Full ControlNet training without needing extreme quantization or low-rank compromises.
* Distributed Data Parallel (DDP) across 2 GPUs with effective batch sizes of 16 to 32.
* Mixed precision (`fp16`) with `use_8bit_adam` and `gradient_checkpointing` for rapid convergence.

---

### 2. Model Architecture Comparison

| Model Architecture | Native Resolution | VRAM during Training | Verdict for Indian Suspect Pipeline |
| :--- | :--- | :--- | :--- |
| **Stable Diffusion v1.5 + ControlNet** | **512 × 512** | **~10–12 GB per GPU** | ⭐ **RECOMMENDED (Best Choice):** Native 1:1 match with your 512×512 dataset. Stable zero-convolution training, fast convergence (~30 mins on 2x 3090), and seamless integration with SD 1.5 Inpainting for human-in-the-loop part refinement. |
| **SDXL 1.0 + ControlNet** | 1024 × 1024 | ~20–22 GB per GPU | **Not Optimal:** SDXL is trained natively at 1024×1024. Training 512px images on SDXL causes aspect/blur artifacts unless upsampled. Significantly slower training (~4–5 hours). |
| **Flux.1 (Schnell / Dev) + ControlNet** | 1024 × 1024 | >24 GB (Requires NF4 QLoRA) | **Too Heavy:** Flux (12B parameters) cannot fit standard ControlNet training in 24GB without 4-bit quantization (QLoRA), has high latency (5–8s per image), and lacks native localized inpainting support for our part masks. |

---

### 3. Fine-Tuning Methodology: Full ControlNet vs. LoRA vs. QLoRA

#### **Why QLoRA is NOT needed for this setup:**
* **QLoRA (Quantized 4-bit LoRA)** was invented for massive LLMs (7B–70B) or 12B diffusion transformers (Flux) that cannot fit into 24GB VRAM in 16-bit.
* **Stable Diffusion 1.5 is only 860M parameters (~3.4 GB in FP16).** It fits comfortably in your 24GB RTX 3090 with over 14GB of free memory to spare.
* Training in pure **FP16** avoids 4-bit quantization noise, ensuring maximum fidelity for fine facial lines and subtle Indian skin tones.

#### **Recommended Hybrid Strategy: Full ControlNet + Text LoRA**
1. **Full ControlNet (Trainable Encoder Copy + Zero Convolutions):**
   * Freezes the pre-trained SD 1.5 UNet backbone to prevent catastrophic forgetting.
   * Trains the ControlNet copy to map sketch lines, wobbles, and shading directly to facial 3D structure.
2. **CLIP Text Encoder LoRA (Optional Rank 16 Adapter):**
   * Injects low-rank adapters into the CLIP text encoder to adapt it to forensic Indian terminology (*"wheatish complexion"*, *"sharp jawline"*, *"thick South-Asian mustache"*).

---

### 4. Training Recipe & Hyperparameters for Dual RTX 3090

Using Hugging Face `diffusers` with `accelerate`:

* **Base Model:** `runwayml/stable-diffusion-v1-5`
* **Precision:** `fp16` (Mixed Precision)
* **Resolution:** `512`
* **Batch Size per GPU:** `8` (Total physical batch size across 2 GPUs = `16`)
* **Gradient Accumulation Steps:** `2` (Effective batch size = `32`)
* **Learning Rate:** `1e-5` (Linear warmup for 500 steps, Cosine decay)
* **Optimizer:** AdamW (`use_8bit_adam=True`)
* **Conditioning Dropout:** `0.05` (Randomly drop text prompt to force the model to rely on the sketch)
* **Epochs:** 15–20 Epochs (~3,000 – 4,000 optimizer steps)
* **Estimated Training Time:** **~30 to 45 minutes on 2× RTX 3090**

#### Execution Command:
```bash
accelerate launch --multi_gpu --num_processes=2 train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./checkpoints/controlnet_indian_faces_512" \
    --dataset_name="dataset_manifest.jsonl" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --max_train_steps=4000 \
    --checkpointing_steps=1000 \
    --validation_image="sketches_augmented/indian_1996_aug1_rough.jpg" \
    --validation_prompt="A middle-aged Indian male in his late 30s with oval face shape, wheatish brown skin tone, gentle dark brown almond eyes..."
```

---

## 🔄 Phase 3: Human-in-the-Loop Refinement (Problem 3 Inpainting)

Once the base face is generated from the sketch, the suspect composite enters the **Refinement Stage**:

```
[Eyewitness Feedback]
"The overall face is correct, but the nose was thinner and sharper."
        │
        ▼
[Automatic Mask Selection]
Load 'Dataset_04_Masks_512/part_masks/indian_xxxx_nose.png'
        │
        ▼
[SD Inpainting Pipeline]
- Frozen Pixels: 100% of the face, eyes, hair, skin, and ears remain unchanged
- Regenerated Pixels: ONLY the nose region is re-drawn with prompt:
  "a sharp, pointed, narrow Indian nose bridge"
        │
        ▼
[Updated 512x512 Suspect Composite]
```

---

## 🖥️ Phase 4: Interactive Web Application (Gradio / Streamlit)

The final deliverable is an intuitive forensic desktop/web interface:
1. **Left Panel (Input Canvas):** Draw a rough sketch or upload an existing image + type witness prompt.
2. **Center Panel (Generated Composite):** High-resolution 512×512 photo rendered in ~1.2s.
3. **Right Panel (Refinement Toolbar):** Interactive buttons:
   * 🔘 `Refine Nose` ➔ Prompts for new nose description.
   * 🔘 `Refine Eyes & Brows` ➔ Prompts for eye adjustments.
   * 🔘 `Refine Mouth & Lips` ➔ Prompts for smile/lip modifications.
   * 🔘 `Refine Hair` ➔ Prompts for hairstyle / hairline changes.
4. **Export:** Export formal forensic composite report with prompt history and audit trail.

---

## 📂 Repository File Structure

```
c:\Users\tanis\OneDrive\Documents\Ai-Ml\IVP\
│
├── 📦 Standardized Training Datasets
│   ├── Dataset_01_Faces_512.zip           (771 MB | 2,000 Restored 512×512 Faces)
│   ├── Dataset_02_Captions.zip            (780 KB | 2,000 Forensic Descriptions)
│   ├── Dataset_03_Sketches_Augmented.zip  (92.8 MB | 6,000 Multi-Style Sketches)
│   └── Dataset_04_Masks_512.zip           (27.9 MB | 12,000 Native 512×512 Part Masks)
│
├── 📦 Source Backups
│   ├── Raw_Faces_224_Source.zip           (20.3 MB | Original 224×224 JPG photos)
│   └── Raw_Sketches_224_Source.zip        (27.7 MB | Original 224×224 JPG sketches)
│
├── 📑 Manifests & Splits
│   ├── dataset_manifest.jsonl             (6,000 paired training lines)
│   ├── dataset_summary.json               (Train: 4,800 | Val: 600 | Test: 600)
│   └── README.md                          (Project documentation & technical roadmap)
│
├── ⚙️ Pipeline Scripts & Modules
│   ├── 01_Face_Super_Resolution_512_Colab.ipynb (Colab GPU CodeFormer notebook)
│   ├── 01_face_super_resolution_colab.py        (Colab GPU CodeFormer script)
│   ├── 03_generate_face_masks_512.py            (512×512 BiSeNet semantic mask parser)
│   ├── 04_build_dataset_manifest.py             (Multi-modal manifest compiler)
│   ├── bisenet_model.py                         (BiSeNet PyTorch neural network module)
│   ├── test_single_image_restoration.py         (Local verification test harness)
│   └── parsing_bisenet.pth                      (50.8 MB BiSeNet weights)
│
└── 🖼️ previews/
    ├── 01_face_restoration_preview.png    (Face restoration verification chart)
    ├── 02_sketch_augmentation_preview.png (Augmented sketch styles chart)
    └── 03_face_parsing_512_preview.png    (512 Face parsing & part masks chart)
```

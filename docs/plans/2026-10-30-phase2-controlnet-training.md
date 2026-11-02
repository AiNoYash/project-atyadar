# Phase 2: Sketch-to-Face ControlNet Fine-Tuning Implementation Plan

> **For AI Assistants & Developers:** Execute this plan task-by-task. All scripts and configs must be ready to transfer and launch on the dual NVIDIA RTX 3090 (2 × 24GB) GPU system.

**Goal:** Train a high-fidelity sketch-conditioned diffusion model (ControlNet + Photorealistic SD 1.5) on the 6,000 multi-style forensic sketch pairs and 2,000 Indian face identities using distributed data parallel (DDP) across 2 × RTX 3090 GPUs.

**Architecture:** Frozen photorealistic Stable Diffusion 1.5 backbone (`SG161222/Realistic_Vision_V5.1_noVAE` or `runwayml/stable-diffusion-v1-5`) + trainable ControlNet encoder with zero-convolutions. Conditioning inputs: 512×512 augmented forensic sketches (clean, rough wobble, faded pencil) and rich forensic text prompts.

**Tech Stack:** PyTorch 2.x, Hugging Face `diffusers` (v0.35+), `accelerate`, `transformers`, `bitsandbytes` (8-bit AdamW), `torchvision`, `Pillow`.

---

## 🏆 Model Architecture Selection & Recommendation

### Comparative Evaluation

| Model Option | Native Resolution | VRAM per GPU (Train) | Convergence Time (2x 3090) | Forensic Suitability | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Realistic Vision v5.1 + ControlNet (SD 1.5)** | **512 × 512** | **~10–12 GB** | **~35–45 mins** | **Exceptional:** Photorealistic skin textures, Indian ethnic undertones, eye glints, natural lighting without cartoonish bias. | ⭐ **PRIMARY CHOICE (Recommended)** |
| **RunwayML Stable Diffusion v1.5 + ControlNet** | 512 × 512 | ~10–12 GB | ~35–45 mins | Good generalist, but faces can sometimes look plastic or airbrushed without specialized negative prompts. | **Secondary / Fallback** |
| **SDXL 1.0 (RealVisXL) + ControlNet** | 1024 × 1024 | ~20–22 GB | ~4–5 hours | High resolution, but requires upscaling 512px dataset to 1024px; tight VRAM headroom; 4x slower training. | **Not Recommended for 512px Dataset** |
| **Flux.1 (Schnell/Dev) + ControlNet** | 1024 × 1024 | >24 GB (Needs 4-bit QLoRA) | Heavy (>6 hours) | DiT architecture lacks native 512px alignment and stable part-mask inpainting for Phase 3. | **Not Recommended** |

### Why Realistic Vision v5.1 (SD 1.5) is the Ideal Model:
1. **1:1 Resolution Match:** The Phase 1 ground-truth faces and sketches are natively 512×512. SD 1.5 operates natively at 512×512 with zero interpolation blur.
2. **Phase 3 Direct Compatibility:** In Phase 3, we will use our 12,000 pre-generated 512×512 part masks (`nose`, `eyes_brows`, `mouth_lips`, `hair`) for selective forensic inpainting. SD 1.5 inpainting models plug in natively with zero latency or sizing mismatches.
3. **Dual RTX 3090 Efficiency:** Dual 24GB GPUs allow FP16 training with batch size 8 per GPU (effective batch size 32 with 2 gradient accumulation steps) and 8-bit AdamW, finishing in ~40 minutes.

---

## 📋 Implementation Tasks Breakdown

### Task 1: Environment & Dependency Setup
**Files:**
- Create: `requirements_train.txt`
- Create: `accelerate_dual_gpu_config.yaml`

**Step 1:** Specify exact pinned versions for training:
- `torch>=2.1.0`
- `torchvision`
- `diffusers>=0.35.0`
- `accelerate>=0.28.0`
- `transformers>=4.38.0`
- `bitsandbytes>=0.42.0`
- `scipy`
- `safetensors`
- `datasets`

**Step 2:** Define multi-GPU DDP configuration with mixed precision `fp16` for `accelerate launch`.

---

### Task 2: Dataset Unpacking & Integrity Verification
**Files:**
- Create: `extract_and_verify_dataset.py`

**Step 1:** Automatically unpack `Dataset_01_Faces_512.zip` into `faces_512/` and `Dataset_03_Sketches_Augmented.zip` into `sketches_augmented/` if directories do not exist.
**Step 2:** Verify that all 6,000 pairs listed in `dataset_manifest.jsonl` exist on disk and have valid 512×512 dimensions.

---

### Task 3: PyTorch Multi-Modal Dataset Module
**Files:**
- Create: `forensic_dataset.py`
- Test: `test_dataset_loader.py`

**Step 1:** Implement `ForensicSketchDataset(Dataset)`:
- Loads `dataset_manifest.jsonl` filtered by split (`train`, `validation`, `test`).
- Reads target face (`image`) normalized to `[-1, 1]`.
- Reads conditioning sketch (`conditioning_image`) normalized to `[0, 1]`.
- Applies classifier-free guidance conditioning dropout (e.g., 5% probability of empty prompt) so the ControlNet learns strong sketch adherence.
- Returns tokenized prompt embeddings via CLIP tokenizer.

---

### Task 4: Complete Distributed Training Script (`train_controlnet.py`)
**Files:**
- Create: `train_controlnet.py`

**Key Features:**
- Model backbone: `SG161222/Realistic_Vision_V5.1_noVAE` (fallback `runwayml/stable-diffusion-v1-5`).
- Instantiates `ControlNetModel.from_unet(unet)` (copies weights from pre-trained UNet down-blocks and mid-block with zero-convolutions).
- Freezes VAE, text encoder, and UNet; enables gradients ONLY on ControlNet.
- Uses `accelerate.Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)`.
- Optimizer: `bitsandbytes.optim.AdamW8bit` at learning rate `1e-5`.
- Validation callback: Evaluates 4 fixed validation sketches every 500 steps, saving preview grids.
- Saves final weights as safetensors compatible with `StableDiffusionControlNetPipeline`.

---

### Task 5: Checkpoint Evaluation & Inference Harness
**Files:**
- Create: `evaluate_controlnet.py`

**Key Features:**
- Loads trained ControlNet checkpoint + base model into `StableDiffusionControlNetPipeline`.
- Evaluates against held-out validation samples from `dataset_manifest.jsonl`.
- Generates side-by-side comparison images: `[Conditioning Sketch | Generated Composite | Ground Truth Face]`.
- Computes structural similarity (SSIM) and perceptual metrics.

---

### Task 6: GPU Execution Runbook & Verification
**Documentation:**
- Quick-start guide for launching on the Dual RTX 3090 system.
- Monitoring GPU VRAM via `nvidia-smi`.
- Resuming from checkpoints.


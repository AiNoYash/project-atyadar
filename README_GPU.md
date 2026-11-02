# 🚀 Phase 2 GPU Training Quickstart (Dual RTX 3090)

## 1. Quick Launch
Unzip this bundle on your GPU workstation. Ensure `Dataset_01_Faces_512.zip` and `Dataset_03_Sketches_Augmented.zip` are in the folder (or extracted as `faces_512` and `sketches_augmented`).

### Linux:
```bash
bash run_train.sh
```

### Windows:
Double-click `run_train.bat` or run:
```cmd
run_train.bat
```

---

## 2. Automated Self-Setup Features:
1. **Self-Audits & Installs Libraries:** Automatically runs `check_env.py` to verify packages and minimum versions (`torch`, `torchvision`, `diffusers`, `accelerate`, `transformers`, `bitsandbytes`, `safetensors`). If any is missing or outdated, it auto-downloads and installs it.
2. **Auto-Detects CUDA & GPUs:** Confirms 2x RTX 3090 (24GB each) and sets up DDP multi-GPU training.
3. **Auto-Extracts Datasets:** Unzips faces and sketches if not already extracted.
4. **Auto-Downloads Base Model:** Automatically pulls `SG161222/Realistic_Vision_V5.1_noVAE` (SD 1.5 photorealistic backbone) on first run from Hugging Face.
5. **Periodic Checkpoints:** Saves ControlNet weights every 500 steps to `./checkpoints/controlnet_indian_faces_512/`.

---

## 3. Testing Inference

### Mode A: Sketch + Text Prompt (Default)
```bash
python infer_controlnet.py \
    --sketch_path="sketches_augmented/indian_0001_orig.jpg" \
    --prompt="An Indian male in his late 20s with sharp jawline, wheatish brown skin tone, dark eyebrows, almond eyes" \
    --output_path="test_suspect.png"
```

### Mode B: Sketch + Text Prompt + Optional Low-Res Photo (CCTV / Mobile)
```bash
python infer_controlnet.py \
    --sketch_path="sketches_augmented/indian_0001_orig.jpg" \
    --prompt="An Indian male with sharp jawline, wheatish brown skin tone, gentle dark eyes" \
    --init_image="path/to/low_res_cctv.jpg" \
    --strength=0.65 \
    --output_path="test_suspect_cctv_enhanced.png"
```

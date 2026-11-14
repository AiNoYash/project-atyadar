# 🎬 Video Presentation Script & Slide Guide
## Project: Multi-Modal Forensic Suspect-Face Composite Generator
**Target Duration:** 4–6 Minutes  
**Assets to Show:** 
1. System Architecture Flowchart: `previews/pipeline_architecture_flowchart.png`
2. Model Comparison Benchmark: `previews/model_comparison_benchmark.png` (or generated in Colab)
3. Live Code / Colab Notebooks: `02_Train_ControlNet_512_Colab.ipynb` & `03_Model_Comparison_Colab.ipynb`

---

### ⏱️ Timeline & Scene Breakdown

| Timestamp | Scene / Visual on Screen | Voiceover Narration & Talking Points |
| :--- | :--- | :--- |
| **0:00 - 0:45** | **Slide 1: Title & The Forensic Problem**<br>*(Show Left Column of Flowchart)* | **"Hello everyone. Today I'm presenting our AI-powered Forensic Suspect Face Composite Generator."**<br><br>• *The Problem:* In criminal investigations, police rely on eyewitness hand-drawn sketches to generate suspect composites.<br>• *Why Existing AI Fails:* Standard text-to-image models (like vanilla Stable Diffusion) have severe **demographic bias**—they default to Caucasian faces and cannot interpret rough pencil sketches accurately, often hallucinating lines as scars or wireframes.<br>• *Our Goal:* Build an end-to-end multi-modal pipeline tailored specifically for **Indian demographics**, turning rough witness sketches and verbal descriptions into 512×512 photorealistic suspect photos. |
| **0:45 - 2:00** | **Slide 2: Phase 1 — Dataset Engineering**<br>*(Show Middle Column of Flowchart)* | **"Phase 1 solves the core data bottleneck: No high-resolution Indian sketch-to-photo dataset existed."**<br><br>1. **Data Ingestion:** We ingested 2,000 Indian face images and initial sketches.<br>2. **CodeFormer Face Super-Resolution:** Raw images were only 224px. We used CodeFormer (VQGAN + Transformer codebooks) to upscale them to **512×512 pore-level ground-truth photos**.<br>3. **Triple Sketch Augmentation:** Real witness sketches aren't clean lines. We algorithmically augmented each sketch into 3 distinct forensic styles: *Original Clean, Rough/Shaky Wobble, and Faded Pencil*—creating **6,000 augmented sketches**.<br>4. **BiSeNet 19-Class Semantic Masking:** Extracted 12,000 pixel-accurate masks for eyes, nose, lips, and hair to prepare for human-in-the-loop editing.<br>5. **Identity-Stratified Manifest:** Compiled `dataset_manifest.jsonl` with an 80/10/10 train/val/test split strictly segregated by identity to prevent data leakage. |
| **2:00 - 3:15** | **Slide 3: Phase 2 — ControlNet Architecture & Training**<br>*(Show Right Column of Flowchart)* | **"In Phase 2, we fine-tuned a ControlNet to condition Stable Diffusion on sketch geometry."**<br><br>• **Model Choice:** We selected `Realistic_Vision_V5.1_noVAE` (SD 1.5 backbone) because it excels in realistic skin lighting and texture, natively matching 512×512 resolution.<br>• **Dual-Branch Architecture:** The base UNet remains frozen to preserve photorealism, while a trainable ControlNet copy learns to inject sketch spatial features via Zero-Convolutions.<br>• **Training Optimization:** Trained on a Tesla T4 GPU using BitsAndBytes **8-bit AdamW** with mixed precision `fp16` and a cosine learning rate scheduler.<br>• **Convergence:** Over 3,000 steps, loss converged smoothly from `0.15` down to `0.012` on fine-detail timesteps, with intermediate checkpoints saved directly to Google Drive. |
| **3:15 - 4:30** | **Slide 4: Live Comparison — Fine-Tuned vs Baseline**<br>*(Show 3-Panel Comparison Figure)* | **"Here is the definitive proof of our model's performance: comparing our fine-tuned model against a standard baseline model using the exact same sketch and prompt."**<br><br>• **Input:** An augmented witness sketch + Prompt: *'Indian male suspect, wheatish skin, defined jawline, neutral forensic gaze'* (Seed 42).<br>• **Baseline Model (Normal SD 1.5):** Notice the failure modes—it generates an ambiguous European face, the skin looks like smooth plastic, and it misinterprets pencil strokes as artifacts.<br>• **Our Fine-Tuned Model:** Delivers authentic Indian ethnic bone structure, natural wheatish skin pigmentation, pore-level texture, specular eye catchlights, and millimeter-level alignment to the sketch's jawline and nose.<br>• *Forensic Value:* Law enforcement gets an actionable, photorealistic composite that witnesses and the public can immediately recognize. |
| **4:30 - 5:00** | **Slide 5: Summary & Next Steps**<br>*(Show Roadmap Summary)* | **"To summarize our progress:"**<br><br>• Phase 1 (Dataset Engineering) & Phase 2 (ControlNet Fine-Tuning) are **100% complete and verified**.<br>• **Next Phase:** Phase 3 introduces **Semantic Part Inpainting**, allowing investigators to selectively swap noses, eyes, or hairstyles based on witness feedback.<br>• Phase 4 packages the entire system into a **Gradio Web App** for police departments.<br><br>**"Thank you for watching!"** |

---

### 💡 Tips for Recording Your Video

1. **Screen Layout:** 
   - Put your camera in the corner (or voiceover only).
   - Display `previews/pipeline_architecture_flowchart.png` in full-screen during the first half.
   - Switch to Google Colab or `model_comparison_benchmark.png` during the second half to demonstrate live AI execution.
2. **Audio:** Speak clearly and emphasize the words **"Demographic Bias"**, **"Multi-Modal Dataset"**, and **"Pore-Level Photorealism"**—these are the key technical discriminators of your project.

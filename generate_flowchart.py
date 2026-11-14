import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_flowchart():
    # 16:9 aspect ratio at high resolution (1920x1080 equivalent at 150-200 dpi)
    fig, ax = plt.subplots(figsize=(20, 11.25), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper function to draw rounded boxes / cards
    def draw_card(x, y, w, h, bg_color, border_color="#334155", radius=1.5, alpha=1.0):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.5,rounding_size={radius}",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            alpha=alpha,
            zorder=2
        )
        ax.add_patch(rect)
        return rect

    # Helper function to draw directional arrows
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", lw=2.2, style="->"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=18,
                shrinkA=3,
                shrinkB=3
            ),
            zorder=5
        )

    # -------------------------------------------------------------
    # 1. HEADER & TITLE
    # -------------------------------------------------------------
    ax.text(50, 96.5, "FORENSIC SUSPECT-FACE COMPOSITE GENERATOR", 
            fontsize=24, fontweight="bold", color="#F8FAFC", ha="center", fontfamily="sans-serif")
    ax.text(50, 93.8, "End-to-End System Architecture: Phase 1 (Dataset Engineering) & Phase 2 (ControlNet Fine-Tuning)", 
            fontsize=13, color="#94A3B8", ha="center", fontfamily="sans-serif")

    # -------------------------------------------------------------
    # 2. COLUMN 1: THE NEED & FORENSIC CHALLENGE (Left)
    # -------------------------------------------------------------
    draw_card(3, 10, 26, 80, bg_color="#111827", border_color="#EF4444")
    
    # Header Badge
    draw_card(5, 83.5, 22, 5, bg_color="#7F1D1D", border_color="#EF4444")
    ax.text(16, 86, "THE FORENSIC PROBLEM & NEED", fontsize=11, fontweight="bold", color="#FEE2E2", ha="center", va="center")

    need_points = [
        ("1. Severe Demographic Bias", "Standard generative models (SD 1.5, SDXL)\nare heavily trained on Western/Caucasian faces,\nresulting in demographic hallucination and\ninability to render authentic Indian facial types."),
        ("2. Forensic Domain Gap", "Witness sketches are drawn by hand with varied\nstyles: rough pencils, shaky strokes, and faded\noutlines. Vanilla models treat sketch lines as\nwires/scars or completely ignore them."),
        ("3. High-Resolution Requirement", "Standard forensic face databases (CUFS, IIITD)\nare only 224x224 and blurry. Modern police\ninvestigations require 512x512 photo-clarity\nwith forensic micro-details (pores, stubble)."),
        ("4. Human-in-the-Loop Need", "Witnesses recall faces iteratively ('narrower nose',\n'change hair'). A system needs paired semantic\nmasks to enable precision part editing (Phase 3).")
    ]

    y_pos = 78
    for title, desc in need_points:
        draw_card(4.5, y_pos - 10, 23, 11, bg_color="#1E293B", border_color="#475569")
        ax.text(6, y_pos - 1.5, title, fontsize=10.5, fontweight="bold", color="#F87171")
        ax.text(6, y_pos - 5.5, desc, fontsize=8.2, color="#CBD5E1", linespacing=1.35)
        y_pos -= 14

    # The Core Objective Card
    draw_card(4.5, 12, 23, 7.5, bg_color="#1E1B4B", border_color="#6366F1")
    ax.text(16, 17, "Core Project Objective", fontsize=10.5, fontweight="bold", color="#A5B4FC", ha="center")
    ax.text(16, 14, "Turn sparse witness sketches into photorealistic\nIndian suspect composite mugshots with exact geometry.", 
            fontsize=8, color="#E0E7FF", ha="center", linespacing=1.3)

    # -------------------------------------------------------------
    # 3. COLUMN 2: PHASE 1 - DATASET ENGINEERING (Middle)
    # -------------------------------------------------------------
    draw_card(32, 10, 32, 80, bg_color="#0F172A", border_color="#0EA5E9")
    
    # Header Badge
    draw_card(34, 83.5, 28, 5, bg_color="#0C4A6E", border_color="#0284C7")
    ax.text(48, 86, "PHASE 1: MULTI-MODAL DATASET CREATION", fontsize=11, fontweight="bold", color="#E0F2FE", ha="center", va="center")

    # Step A: Raw Input
    draw_card(33.5, 71.5, 29, 9, bg_color="#1E293B", border_color="#38BDF8")
    ax.text(35, 78, "Step 1.1: Raw Data Ingestion (224px)", fontsize=10, fontweight="bold", color="#38BDF8")
    ax.text(35, 74.5, "Input: 2,000 raw Indian faces + initial sketches.\nInitial state: Low resolution (224x224), heavy JPEG artifacts.", fontsize=8, color="#94A3B8")

    draw_arrow(48, 71.5, 48, 67.5, color="#38BDF8")

    # Step B: CodeFormer Face Restoration
    draw_card(33.5, 58, 29, 9, bg_color="#1E293B", border_color="#38BDF8")
    ax.text(35, 64.5, "Step 1.2: CodeFormer Super-Resolution (512px)", fontsize=10, fontweight="bold", color="#38BDF8")
    ax.text(35, 61, "VQGAN + Transformer codebook restoration.\nResult: 2,000 pore-level 512x512 ground-truth Indian photos.", fontsize=8, color="#94A3B8")

    draw_arrow(48, 58, 48, 54, color="#38BDF8")

    # Step C: Parallel Triple Augmentation & BiSeNet Masking
    # Sketch Augmentation Card
    draw_card(33.5, 41, 14, 12.5, bg_color="#1E293B", border_color="#06B6D4")
    ax.text(34.5, 51.5, "Step 1.3: Sketch Augmentation", fontsize=8.5, fontweight="bold", color="#22D3EE")
    ax.text(34.5, 45.5, "- Style 1: Original Clean\n- Style 2: Rough Wobble\n- Style 3: Faded Pencil\nResult: 6,000 Augmented\nSketches (simulates witnesses)", fontsize=7.5, color="#CBD5E1", linespacing=1.25)

    # Face Parsing Card
    draw_card(48.5, 41, 14, 12.5, bg_color="#1E293B", border_color="#06B6D4")
    ax.text(49.5, 51.5, "Step 1.4: BiSeNet Parsing", fontsize=8.5, fontweight="bold", color="#22D3EE")
    ax.text(49.5, 45.5, "- 19-class facial masks\n- Eyes, nose, lips, hair,\n  skin, brows segmented\nResult: 12,000 masks for\nPhase 3 inpainting", fontsize=7.5, color="#CBD5E1", linespacing=1.25)

    draw_arrow(40.5, 41, 44, 36.5, color="#06B6D4")
    draw_arrow(55.5, 41, 52, 36.5, color="#06B6D4")

    # Step D: Manifest Compilation
    draw_card(33.5, 25, 29, 11, bg_color="#1E293B", border_color="#38BDF8")
    ax.text(35, 33.5, "Step 1.5: Multi-Modal Manifest & Stratification", fontsize=10, fontweight="bold", color="#38BDF8")
    ax.text(35, 28, "- Strict identity split to prevent leakage:\n  80% Train (4,800 pairs) | 10% Val | 10% Test\n- Paired: 512px Face + 3 Sketches + Captions + Masks\nOutput: dataset_manifest.jsonl (Master Registry)", fontsize=8, color="#CBD5E1", linespacing=1.3)

    draw_arrow(48, 25, 48, 20.5, color="#38BDF8")

    # Step E: Phase 1 Final Package
    draw_card(33.5, 12, 29, 8, bg_color="#064E3B", border_color="#10B981")
    ax.text(48, 17, "PHASE 1 COMPLETE: Verified Multi-Modal Dataset", fontsize=9.5, fontweight="bold", color="#6EE7B7", ha="center")
    ax.text(48, 14, "2,000 512px Faces | 6,000 Sketches | 12,000 Masks | Captions", fontsize=8, color="#D1FAE5", ha="center")

    # Arrow from Phase 1 to Phase 2
    draw_arrow(64, 48, 67, 48, color="#10B981", lw=3.0)

    # -------------------------------------------------------------
    # 4. COLUMN 3: PHASE 2 - CONTROLNET FINE-TUNING (Right)
    # -------------------------------------------------------------
    draw_card(67, 10, 30, 80, bg_color="#0F172A", border_color="#10B981")
    
    # Header Badge
    draw_card(68.5, 83.5, 27, 5, bg_color="#064E3B", border_color="#10B981")
    ax.text(82, 86, "PHASE 2: CONTROLNET FINE-TUNING", fontsize=11, fontweight="bold", color="#D1FAE5", ha="center", va="center")

    # Architecture Overview
    draw_card(68.5, 70, 27, 11.5, bg_color="#1E293B", border_color="#34D399")
    ax.text(70, 79, "Model Architecture: Dual-Branch Network", fontsize=9.5, fontweight="bold", color="#34D399")
    ax.text(70, 73, "- Backbone: SG161222/Realistic_Vision_V5.1 (Frozen UNet)\n  Guarantees realistic skin, lighting, and texture.\n- ControlNet Branch: Trainable copy of UNet encoder\n  connected via Zero-Convolutions for sketch control.", fontsize=7.8, color="#CBD5E1", linespacing=1.3)

    draw_arrow(82, 70, 82, 66, color="#34D399")

    # Multi-Modal Conditioning Inputs
    draw_card(68.5, 53, 27, 12.5, bg_color="#1E293B", border_color="#34D399")
    ax.text(70, 63, "Multi-Modal Conditionings During Training", fontsize=9.5, fontweight="bold", color="#34D399")
    ax.text(70, 56.5, "1. Spatial: 512x512 Augmented Sketch (Geometry)\n2. Semantic: Forensic Text Prompt via CLIP\n   ('Indian male, sharp jawline, wheatish skin...')\n3. Optional Prior: Low-Res/CCTV photo (--init_image)", fontsize=7.8, color="#CBD5E1", linespacing=1.3)

    draw_arrow(82, 53, 82, 49, color="#34D399")

    # Training Specifications
    draw_card(68.5, 36, 27, 12.5, bg_color="#1E293B", border_color="#34D399")
    ax.text(70, 46, "Hardware & Training Optimization", fontsize=9.5, fontweight="bold", color="#34D399")
    ax.text(70, 39.5, "- Hardware: Tesla T4 (15GB) / Dual RTX 3090 (24GB)\n- Optimizer: BitsAndBytes 8-bit AdamW (VRAM saving)\n- Precision: Mixed precision fp16 | Batch: 4 | Accum: 4\n- Total Steps: 3,000 steps (~10 epochs, Cosine decay)", fontsize=7.8, color="#CBD5E1", linespacing=1.3)

    draw_arrow(82, 36, 82, 32, color="#34D399")

    # Final Convergence & Result
    draw_card(68.5, 12, 27, 19.5, bg_color="#1E293B", border_color="#F59E0B")
    ax.text(70, 29, "Convergence & Trained Artifacts", fontsize=9.5, fontweight="bold", color="#FBBF24")
    ax.text(70, 22.5, "- Loss Convergence: Smooth decline from 0.15 -> 0.012\n- Google Drive Checkpointing: checkpoint-500 .. 2500\n- Final Artifact: final_controlnet (1.45 GB weights)\n- Inference: DPM-Solver++ (25 steps, ~3.5s per composite)", fontsize=7.8, color="#CBD5E1", linespacing=1.35)
    
    # Status Banner
    draw_card(70, 13.5, 24, 6, bg_color="#78350F", border_color="#F59E0B")
    ax.text(82, 16.5, "PHASE 2 COMPLETE: Production Model Verified!", fontsize=8.5, fontweight="bold", color="#FEF3C7", ha="center", va="center")

    plt.tight_layout()
    output_path = Path("previews/pipeline_architecture_flowchart.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0B0F19")
    plt.close()
    print(f"[SUCCESS] Flowchart image saved to: {output_path}")

if __name__ == "__main__":
    draw_flowchart()

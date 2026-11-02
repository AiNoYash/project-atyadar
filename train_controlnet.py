"""
train_controlnet.py - Forensic Sketch-to-Face ControlNet Training Pipeline
Trained on 512x512 Indian Suspect multi-modal dataset using Diffusers & Accelerate.
Supports 1x or 2x NVIDIA RTX 3090 GPUs (DDP) and CPU dry-run verification.
"""

import os
import json
import zipfile
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup

logger = get_logger(__name__)


def auto_extract_datasets(base_dir: Path):
    """Auto-extracts dataset zips to fast local disk if target folders do not exist."""
    # Check possible source zip locations (local or Colab Google Drive)
    search_dirs = [base_dir, Path("/content/drive/MyDrive/IVP"), Path("/content")]
    
    # Target extraction directory (prefer /content for fast NVMe in Colab, otherwise base_dir)
    target_root = Path("/content") if Path("/content").exists() and not (base_dir / "faces_512").exists() else base_dir

    # 1. Faces 512
    faces_dir = target_root / "faces_512"
    faces_alt = base_dir / "faces_512"
    if not faces_dir.exists() and not faces_alt.exists():
        for d in search_dirs:
            for fname in ["Dataset_01_Faces_512.zip", "Indian_Face_512.zip"]:
                fz = d / fname
                if fz.exists():
                    print(f"[EXTRACT] Extracting {fz} to fast disk {faces_dir}...")
                    faces_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(fz, "r") as z:
                        z.extractall(faces_dir)
                    print("[OK] Faces extraction complete.")
                    break
            if faces_dir.exists():
                break

    # 2. Sketches Augmented
    sketches_dir = target_root / "sketches_augmented"
    sketches_alt = base_dir / "Indian_Face_Sketches_Augmented"
    if not sketches_dir.exists() and not (base_dir / "sketches_augmented").exists() and not sketches_alt.exists():
        for d in search_dirs:
            for fname in ["Dataset_03_Sketches_Augmented.zip", "Indian_Face_Sketches_Augmented.zip"]:
                sz = d / fname
                if sz.exists():
                    print(f"[EXTRACT] Extracting {sz} to fast disk {sketches_dir}...")
                    sketches_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(sz, "r") as z:
                        z.extractall(sketches_dir)
                    print("[OK] Sketches extraction complete.")
                    break
            if sketches_dir.exists():
                break


class ForensicDataset(Dataset):
    """Loads 512x512 ground-truth faces and conditioning sketches paired via manifest."""

    def __init__(self, manifest_path: str, tokenizer, split: str = "train", cond_dropout: float = 0.05):
        self.base_dir = Path(manifest_path).parent.resolve()
        auto_extract_datasets(self.base_dir)

        self.tokenizer = tokenizer
        self.cond_dropout = cond_dropout
        self.entries = []

        # Candidate root search paths
        self.search_roots = [
            self.base_dir,
            Path("/content"),
            Path("/content/drive/MyDrive/IVP")
        ]

        # Target image transform -> [-1, 1]
        self.img_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Conditioning sketch transform -> [0, 1]
        self.sketch_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        # Manifest search
        actual_manifest = Path(manifest_path)
        if not actual_manifest.exists():
            for root in self.search_roots:
                alt = root / manifest_path
                if alt.exists():
                    actual_manifest = alt
                    break

        with open(actual_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if split == "all" or data.get("split") == split:
                    self.entries.append(data)

        if not self.entries:
            raise ValueError(f"No samples found in {actual_manifest} for split '{split}'")

    def __len__(self):
        return len(self.entries)

    def _resolve_path(self, rel_path: str) -> Path:
        for root in self.search_roots:
            p = root / rel_path
            if p.exists():
                return p
            # Check alternate folder naming
            if "sketches_augmented" in rel_path:
                alt = root / rel_path.replace("sketches_augmented", "Indian_Face_Sketches_Augmented")
                if alt.exists():
                    return alt
        return self.base_dir / rel_path

    def __getitem__(self, idx):
        item = self.entries[idx]

        # 1. Target Face Image
        target_path = self._resolve_path(item["target_image"])
        if target_path.exists():
            target_img = Image.open(target_path).convert("RGB")
        else:
            # Fallback placeholder for missing image during dry-runs
            target_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        pixel_values = self.img_transforms(target_img)

        # 2. Conditioning Sketch
        sketch_path = self._resolve_path(item["conditioning_sketch"])
        if sketch_path.exists():
            sketch_img = Image.open(sketch_path).convert("RGB")
        else:
            sketch_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
        conditioning_pixel_values = self.sketch_transforms(sketch_img)

        # 3. Prompt Conditioning with CFG Dropout
        prompt = item.get("prompt", "")
        if self.cond_dropout > 0 and torch.rand(1).item() < self.cond_dropout:
            prompt = ""

        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for Forensic Sketch-to-Face (Colab T4 & Dual 3090)")
    
    # Auto-detect Google Drive for checkpoints
    colab_drive = Path("/content/drive/MyDrive/IVP/checkpoints")
    default_output = str(colab_drive) if Path("/content/drive/MyDrive").exists() else "./checkpoints/controlnet_indian_faces_512"

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE",
                        help="HuggingFace model ID or path. Fallback: runwayml/stable-diffusion-v1-5")
    parser.add_argument("--manifest_path", type=str, default="dataset_manifest.jsonl",
                        help="Path to dataset_manifest.jsonl")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for saved checkpoints (defaults to Drive if on Colab)")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size per GPU (4 is optimal for Colab T4 16GB, 8 for RTX 3090 24GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = 16 on 1 GPU, 32 on 2 GPUs)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="ControlNet learning rate (default: 1e-5)")
    parser.add_argument("--max_train_steps", type=int, default=3000,
                        help="Total training steps (~10-12 epochs for 4,800 train samples)")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save checkpoint every X steps to Google Drive / local disk")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest",
                        help="Path to checkpoint or 'latest' to auto-resume from highest checkpoint in output_dir")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 2 quick steps on CPU/GPU to verify pipeline integrity")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize Accelerator for Multi-GPU DDP
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if torch.cuda.is_available() else "no",
        project_dir=args.output_dir
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INIT] Initialized Accelerator | Device: {accelerator.device} | Num Processes: {accelerator.num_processes}")

    # 2. Self-downloading Tokenizer & Models
    model_id = args.pretrained_model_name_or_path
    if accelerator.is_main_process:
        print(f"[DOWNLOAD] Loading model from: {model_id} (auto-downloads from HF if not cached)")

    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    except Exception as e:
        if "SG161222" in model_id:
            print(f"[WARN] Primary model {model_id} error: {e}. Falling back to runwayml/stable-diffusion-v1-5...")
            model_id = "runwayml/stable-diffusion-v1-5"
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        else:
            raise e

    # 3. Instantiate or Resume Trainable ControlNet
    resume_step = 0
    if args.resume_from_checkpoint and os.path.exists(args.output_dir):
        if args.resume_from_checkpoint == "latest":
            ckpts = [d for d in Path(args.output_dir).glob("checkpoint-*") if d.is_dir()]
            if ckpts:
                # Sort numerically by step
                ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)
                latest_ckpt = ckpts[-1]
                resume_step = int(latest_ckpt.name.split("-")[-1])
                if accelerator.is_main_process:
                    print(f"[RESUME] Found existing checkpoint: {latest_ckpt.name}. Resuming from step {resume_step}...")
                controlnet = ControlNetModel.from_pretrained(str(latest_ckpt))
            else:
                if accelerator.is_main_process:
                    print("[INIT] Cloning UNet encoder blocks into new ControlNet...")
                controlnet = ControlNetModel.from_unet(unet)
        elif os.path.isdir(args.resume_from_checkpoint):
            if accelerator.is_main_process:
                print(f"[RESUME] Loading weights from specified checkpoint: {args.resume_from_checkpoint}...")
            controlnet = ControlNetModel.from_pretrained(args.resume_from_checkpoint)
            step_part = Path(args.resume_from_checkpoint).name.split("-")[-1]
            resume_step = int(step_part) if step_part.isdigit() else 0
        else:
            if accelerator.is_main_process:
                print("[INIT] Cloning UNet encoder blocks into new ControlNet...")
            controlnet = ControlNetModel.from_unet(unet)
    else:
        if accelerator.is_main_process:
            print("[INIT] Cloning UNet encoder blocks into new ControlNet...")
        controlnet = ControlNetModel.from_unet(unet)

    # 4. Freeze base models (UNet, VAE, Text Encoder) - only train ControlNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)
    controlnet.train()

    # Enable gradient checkpointing on ControlNet to save VRAM
    if hasattr(controlnet, "enable_gradient_checkpointing"):
        controlnet.enable_gradient_checkpointing()

    # 5. Optimizer: 8-bit AdamW for dual RTX 3090, standard AdamW fallback
    learning_rate = args.learning_rate
    optimizer_cls = torch.optim.AdamW
    try:
        if torch.cuda.is_available():
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            if accelerator.is_main_process:
                print("[OPTIM] Using BitsAndBytes 8-bit AdamW optimizer.")
    except Exception:
        if accelerator.is_main_process:
            print("[OPTIM] BitsAndBytes not available; using PyTorch native AdamW.")

    optimizer = optimizer_cls(controlnet.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

    # 6. Dataset & DataLoader
    train_split = "train" if not args.dry_run else "all"
    train_dataset = ForensicDataset(manifest_path=args.manifest_path, tokenizer=tokenizer, split=train_split)
    train_batch_size = 1 if args.dry_run else args.train_batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0 if args.dry_run or os.name == "nt" else min(2, os.cpu_count() or 2),
        pin_memory=torch.cuda.is_available()
    )

    # 7. Learning Rate Scheduler
    max_steps = 2 if args.dry_run else args.max_train_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1 if args.dry_run else 300,
        num_training_steps=max_steps
    )

    # 8. Prepare with Accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # Fast-forward lr_scheduler if resuming from an intermediate checkpoint
    if resume_step > 0:
        for _ in range(resume_step):
            lr_scheduler.step()
        if accelerator.is_main_process:
            print(f"[RESUME] Fast-forwarded LR scheduler to step {resume_step}. Current LR: {lr_scheduler.get_last_lr()[0]:.2e}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # 9. Training Loop
    if accelerator.is_main_process:
        print(f"\n[TRAIN] Starting training: Max Steps={max_steps} | Mixed Precision={accelerator.mixed_precision}")
        if args.dry_run:
            print("[TEST] DRY-RUN MODE: Verifying 2 steps on CPU/GPU...")

    global_step = resume_step
    if resume_step > 0 and accelerator.is_main_process:
        print(f"[RESUME] Fast-forwarding: starting training from step {global_step}/{max_steps}...")
    data_iter = iter(train_dataloader)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(controlnet):
            # A. Encode target images into latent space
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # B. Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # C. Encode text prompt
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # D. ControlNet forward pass with sketch conditioning
            conditioning_sketch = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning_sketch,
                return_dict=False
            )

            # E. UNet forward pass with ControlNet residuals
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                return_dict=False
            )[0]

            # F. Loss computation (predict noise / epsilon)
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            if accelerator.is_main_process:
                lr = lr_scheduler.get_last_lr()[0]
                print(f"Step [{global_step}/{max_steps}] | Loss: {loss.item():.4f} | LR: {lr:.2e}")

                # Save intermediate checkpoints
                if global_step % args.checkpointing_steps == 0 or global_step == max_steps:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    unwrapped = accelerator.unwrap_model(controlnet)
                    unwrapped.save_pretrained(save_path)
                    print(f"[SAVE] Saved ControlNet checkpoint to {save_path}")

    # 10. Save Final ControlNet Model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_controlnet")
        unwrapped = accelerator.unwrap_model(controlnet)
        unwrapped.save_pretrained(final_dir)
        print(f"\n[DONE] Training complete! Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()

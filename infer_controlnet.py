"""
infer_controlnet.py - Forensic Sketch-to-Face Inference
Generates photorealistic 512x512 face composite from an input sketch and text prompt.
Supports optional low-resolution CCTV/mobile photo conditioning (img2img mode).
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Forensic Suspect Face from Sketch")
    parser.add_argument("--controlnet_path", type=str, default="./checkpoints/controlnet_indian_faces_512/final_controlnet",
                        help="Path to trained ControlNet checkpoint")
    parser.add_argument("--base_model", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE",
                        help="Base SD 1.5 photorealistic model ID")
    parser.add_argument("--sketch_path", type=str, required=True,
                        help="Path to input sketch image (512x512 PNG/JPG)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Forensic witness text prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, low quality, cartoon, 3d render, distorted, disfigured, drawing, sketch",
                        help="Negative prompt for realistic output")
    parser.add_argument("--init_image", type=str, default=None,
                        help="Optional: low-resolution CCTV/photo (img2img mode)")
    parser.add_argument("--strength", type=float, default=0.65,
                        help="Denoising strength when using --init_image (0.0 to 1.0, recommended 0.60-0.75)")
    parser.add_argument("--output_path", type=str, default="generated_suspect.png",
                        help="Output image save path")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.85,
                        help="Weight of sketch conditioning (0.0 - 1.0, recommended 0.8-0.9)")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Denoising steps (20-30 is fast and optimal)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[LOAD] Loading ControlNet from: {args.controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=weight_dtype)

    sketch = Image.open(args.sketch_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.init_image is not None and Path(args.init_image).exists():
        print(f"[MODE] Img2Img with Low-Res Photo Conditioning: '{args.init_image}' (strength={args.strength})")
        init_img = Image.open(args.init_image).convert("RGB").resize((512, 512))

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.base_model,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
            safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        pipe.to(device)

        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=init_img,
            control_image=sketch,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator
        ).images[0]
    else:
        print("[MODE] Text-Guided Sketch Conditioning (from pure noise)")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base_model,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
            safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        pipe.to(device)

        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=sketch,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator
        ).images[0]

    image.save(args.output_path)
    print(f"[DONE] Saved generated composite to: {args.output_path}")

if __name__ == "__main__":
    main()

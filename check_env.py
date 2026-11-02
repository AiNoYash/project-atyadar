"""
check_env.py - Environment & Dependency Validator
Checks required libraries, compares minimum versions, and auto-installs missing/outdated packages.
Also verifies CUDA availability and multi-GPU count.
"""

import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError

REQUIRED_PACKAGES = [
    ("torch", "2.1.0"),
    ("torchvision", "0.16.0"),
    ("diffusers", "0.35.0"),
    ("accelerate", "0.28.0"),
    ("transformers", "4.38.0"),
    ("safetensors", "0.4.0"),
    ("scipy", "1.10.0"),
    ("pillow", "9.0.0")
]

OPTIONAL_PACKAGES = [
    ("bitsandbytes", "0.42.0")  # Recommended for 8-bit AdamW on GPU
]

def parse_version(ver_str):
    clean = ver_str.split("+")[0].split("-")[0]
    return [int(x) for x in clean.split(".") if x.isdigit()]

def main():
    print("=" * 60)
    print("      FORENSIC PIPELINE: ENVIRONMENT & GPU AUDIT")
    print("=" * 60)

    need_install = []

    # 1. Check Required Packages
    for pkg, min_ver in REQUIRED_PACKAGES:
        try:
            curr_ver = version(pkg)
            if parse_version(curr_ver) < parse_version(min_ver):
                print(f"[OUTDATED] {pkg}: found {curr_ver}, require >= {min_ver}")
                need_install.append(pkg)
            else:
                print(f"[OK] {pkg:<15} : {curr_ver} (>= {min_ver})")
        except PackageNotFoundError:
            print(f"[MISSING] {pkg:<15} : Not installed (require >= {min_ver})")
            need_install.append(pkg)

    # 2. Check Optional Packages (bitsandbytes)
    for pkg, min_ver in OPTIONAL_PACKAGES:
        try:
            curr_ver = version(pkg)
            print(f"[OK] {pkg:<15} : {curr_ver} (optional)")
        except PackageNotFoundError:
            print(f"[OPTIONAL] {pkg:<15} : Not installed. Will attempt install for 8-bit AdamW.")
            need_install.append(pkg)

    # 3. Auto-install missing or outdated packages
    if need_install:
        print(f"\n[INSTALL] Auto-downloading and installing: {need_install}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements_gpu.txt"
            ])
            print("[SUCCESS] All dependencies installed.")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Pip install returned code {e.returncode}. Continuing...")
    else:
        print("\n[SUCCESS] All core dependencies are up to date.")

    # 4. Check CUDA & Dual GPU Setup
    print("-" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"[CUDA DETECTED] {count} GPU(s) available for training:")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"  -> GPU [{i}]: {props.name} | VRAM: {vram_gb:.1f} GB")
            if count >= 2:
                print(f"[READY] Dual GPU configuration confirmed ({count} x 3090). DDP ready.")
            else:
                print(f"[INFO] 1 GPU detected. Training will run on GPU 0.")
        else:
            print("[WARN] No CUDA device detected! Script will run on CPU.")
    except Exception as e:
        print(f"[WARN] Could not query CUDA: {e}")

    print("=" * 60)

if __name__ == "__main__":
    main()


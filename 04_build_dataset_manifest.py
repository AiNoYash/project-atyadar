"""
==============================================================================
Suspect-Face Composite Generator: Multi-Modal Dataset Manifest Builder
Improvement 2: Pairs 512x512 Faces, Augmented Sketches, Captions & Masks
Generates unified dataset_manifest.jsonl with strict identity train/val/test splits
==============================================================================
"""

import os
import sys
import json
import zipfile
import random
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(candidates):
    for c in candidates:
        p = os.path.join(BASE_DIR, c)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, candidates[0])

ZIP_FACES_512 = find_file(["Dataset_01_Faces_512.zip", "Indian_Face_512.zip"])
ZIP_CAPTIONS = find_file(["Dataset_02_Captions.zip", "Indian_Face_Captions-20261023T051021Z-1-001.zip"])
ZIP_SKETCHES = find_file(["Dataset_03_Sketches_Augmented.zip", "Indian_Face_Sketches_Augmented.zip"])
ZIP_MASKS_512 = find_file(["Dataset_04_Masks_512.zip", "Indian_Face_Masks_512.zip"])


OUTPUT_JSONL = os.path.join(BASE_DIR, "dataset_manifest.jsonl")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "dataset_summary.json")


def build_manifest():
    print("=" * 75)
    print("   SUSPECT-FACE COMPOSITE PIPELINE: UNIFIED DATASET MANIFEST")
    print("=" * 75)

    # 1. Load Captions
    print("[1/5] Loading captions from zip...")
    captions = {}
    with zipfile.ZipFile(ZIP_CAPTIONS, 'r') as z_cap:
        for f in z_cap.namelist():
            if f.endswith('.txt'):
                base = os.path.splitext(os.path.basename(f))[0]
                text = z_cap.read(f).decode('utf-8').strip()
                captions[base] = text
    print(f"      Loaded {len(captions)} captions.")

    # 2. Load Sketches
    print("\n[2/5] Indexing augmented sketches...")
    sketches = {}
    with zipfile.ZipFile(ZIP_SKETCHES, 'r') as z_sk:
        for f in z_sk.namelist():
            if f.lower().endswith(('.jpg', '.png')):
                fname = os.path.basename(f)
                # Formats: indian_0001_orig.jpg, indian_0001_aug1_rough.jpg, indian_0001_aug2_faded.jpg
                # Extract root identity
                parts = fname.split('_')
                if len(parts) >= 2:
                    base_id = f"{parts[0]}_{parts[1]}"
                    if base_id not in sketches:
                        sketches[base_id] = []
                    sketches[base_id].append(fname)
    print(f"      Indexed {sum(len(v) for v in sketches.values())} sketches across {len(sketches)} identities.")

    # 3. Load 512x512 Face Images
    print("\n[3/5] Indexing 512x512 restored faces...")
    faces = {}
    with zipfile.ZipFile(ZIP_FACES_512, 'r') as z_f:
        for f in z_f.namelist():
            if f.lower().endswith(('.png', '.jpg')):
                fname = os.path.basename(f)
                base = os.path.splitext(fname)[0]
                faces[base] = fname
    print(f"      Indexed {len(faces)} 512x512 faces.")

    # 4. Identity Split (Strictly identity-stratified to prevent data leakage)
    print("\n[4/5] Partitioning identities into Train (80%), Val (10%), Test (10%)...")
    all_identities = sorted(list(faces.keys()))
    random.seed(42)  # Deterministic reproducible split
    shuffled_ids = all_identities.copy()
    random.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * 0.80)  # 1,600
    n_val = int(n_total * 0.10)    # 200
    n_test = n_total - n_train - n_val  # 200

    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])

    print(f"      Train Identities: {len(train_ids)} ({len(train_ids)*3} sketch-photo pairs)")
    print(f"      Val Identities:   {len(val_ids)} ({len(val_ids)*3} sketch-photo pairs)")
    print(f"      Test Identities:  {len(test_ids)} ({len(test_ids)*3} sketch-photo pairs)")

    # 5. Build Unified Manifest Rows
    print(f"\n[5/5] Compiling {OUTPUT_JSONL}...")
    manifest_rows = []
    
    for ident in all_identities:
        split = "train" if ident in train_ids else ("val" if ident in val_ids else "test")
        prompt = captions.get(ident, "")
        target_img = faces.get(ident, f"{ident}.png")
        id_sketches = sketches.get(ident, [])

        # For each augmented sketch variant, create a paired training entry
        for sk in id_sketches:
            style = "original" if "orig" in sk else ("rough_wobble" if "aug1" in sk or "rough" in sk else "faded_pencil")
            row = OrderedDict([
                ("id", ident),
                ("split", split),
                ("style", style),
                ("target_image", f"faces_512/{target_img}"),
                ("conditioning_sketch", f"sketches_augmented/{sk}"),
                ("prompt", prompt),
                ("masks", {
                    "raw_19class": f"masks_512/raw_masks/{ident}_mask.png",
                    "nose": f"masks_512/part_masks/{ident}_nose.png",
                    "eyes_brows": f"masks_512/part_masks/{ident}_eyes_brows.png",
                    "mouth_lips": f"masks_512/part_masks/{ident}_mouth_lips.png",
                    "hair": f"masks_512/part_masks/{ident}_hair.png",
                    "face_skin": f"masks_512/part_masks/{ident}_face_skin.png",
                })
            ])
            manifest_rows.append(row)

    # Write JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for r in manifest_rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Summary Statistics
    summary = {
        "dataset_name": "Indian_Suspect_Face_Composite_Dataset",
        "resolution": "512x512",
        "total_identities": n_total,
        "total_paired_samples": len(manifest_rows),
        "splits": {
            "train": {
                "identities": len(train_ids),
                "samples": sum(1 for r in manifest_rows if r["split"] == "train")
            },
            "validation": {
                "identities": len(val_ids),
                "samples": sum(1 for r in manifest_rows if r["split"] == "val")
            },
            "test": {
                "identities": len(test_ids),
                "samples": sum(1 for r in manifest_rows if r["split"] == "test")
            }
        },
        "sketch_styles": {
            "original": sum(1 for r in manifest_rows if r["style"] == "original"),
            "rough_wobble": sum(1 for r in manifest_rows if r["style"] == "rough_wobble"),
            "faded_pencil": sum(1 for r in manifest_rows if r["style"] == "faded_pencil")
        },
        "modalities_included": [
            "target_image_512x512",
            "conditioning_sketch_augmented",
            "forensic_text_prompt",
            "raw_19class_semantic_mask",
            "nose_part_mask",
            "eyes_brows_part_mask",
            "mouth_lips_part_mask",
            "hair_part_mask",
            "face_skin_part_mask"
        ]
    }

    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f_sum:
        json.dump(summary, f_sum, indent=2)

    print(f"[SUCCESS] Wrote {len(manifest_rows)} paired entries to: {os.path.basename(OUTPUT_JSONL)}")
    print(f"[SUCCESS] Wrote summary statistics to: {os.path.basename(OUTPUT_SUMMARY)}")
    print("=" * 75)
    print("  MULTI-MODAL DATASET MANIFEST COMPLETE & PRODUCTION-READY!")
    print("=" * 75)


if __name__ == "__main__":
    build_manifest()

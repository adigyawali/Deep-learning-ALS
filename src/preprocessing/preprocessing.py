import os
import re
import ants
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Output folder naming ──────────────────────────────────────────────────────

def derive_folder_name(t1_path: Path, index: int) -> str:
    """
    Derive a clean output folder name from the T1 filename.
    Strips known suffixes and modality tokens to get e.g. 'P033_V1'.

    Falls back to 'case_{index:04d}' if no subject ID can be found,
    so processing never silently stops on an unusual filename.
    """
    stem = t1_path.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Remove known trailing tokens that are not part of the identity
    for token in ("_synthstrip", "_T1W", "_T1", "_run-01", "_run-02",
                  "_run-1", "_run-2", "_run-03", "_run-3"):
        stem = stem.replace(token, "")

    # Extract subject (C/P + digits) and visit (V + digits) using regex
    subj_match  = re.search(r'(?<![A-Za-z0-9])(C\d+|P\d+)(?![A-Za-z0-9])', stem)
    visit_match = re.search(r'(?<![A-Za-z0-9])(V\d+)(?![A-Za-z0-9])',       stem, re.IGNORECASE)

    if subj_match and visit_match:
        return f"{subj_match.group(1)}_{visit_match.group(1).upper()}"

    if subj_match:
        return f"{subj_match.group(1)}_V1"

    # Absolute fallback — never block processing
    return f"case_{index:04d}"


# ── QC snapshot ───────────────────────────────────────────────────────────────

def generate_qc_snapshot(t1, t2, flair, filename: Path) -> None:
    """Saves a QC image showing the middle axial slice of each modality."""
    def mid_slice(img):
        if img is None:
            return np.zeros((100, 100))
        arr = img.numpy()
        return np.rot90(arr[:, :, arr.shape[2] // 2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mid_slice(t1),    cmap="gray"); axes[0].set_title("T1")
    axes[1].imshow(mid_slice(t2),    cmap="gray"); axes[1].set_title("T2")
    axes[2].imshow(mid_slice(flair), cmap="gray"); axes[2].set_title("FLAIR")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()


# ── Already-processed check ───────────────────────────────────────────────────

def is_already_processed(folder_name: str, output_dir: Path) -> bool:
    """
    Returns True only when all three output NIfTI files already exist
    for this folder name.  Checks the exact files that process_case() saves.
    """
    out = output_dir / folder_name
    if not out.exists():
        return False
    return all((out / f"{folder_name}_{mod}.nii.gz").exists()
               for mod in ("T1", "T2", "FLAIR"))


# ── Per-case processing ───────────────────────────────────────────────────────

def process_case(folder_name: str,
                 t1_path: Path, t2_path: Path, flair_path: Path,
                 output_dir: Path, qc_dir: Path) -> None:
    """
    For one matched triplet:
      1. Reorient to RAS
      2. N4 Bias Field Correction
      3. Register T1 -> MNI152 (Affine)
      4. Register T2 -> T1, apply combined T1->MNI transform
      5. Register FLAIR -> T1, apply combined T1->MNI transform
      6. Save outputs + QC snapshot
    """
    out_dir = output_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{folder_name}] ===")
    print(f"   T1    : {t1_path.name}")
    print(f"   T2    : {t2_path.name}")
    print(f"   FLAIR : {flair_path.name}")

    try:
        # Load
        t1    = ants.image_read(str(t1_path))
        t2    = ants.image_read(str(t2_path))
        flair = ants.image_read(str(flair_path))

        # Reorient
        print("   -> Reorienting to RAS...")
        t1    = ants.reorient_image2(t1,    orientation="RAS")
        t2    = ants.reorient_image2(t2,    orientation="RAS")
        flair = ants.reorient_image2(flair, orientation="RAS")

        # N4 Bias Correction
        print("   -> N4 Bias Correction...")
        t1    = ants.n4_bias_field_correction(t1)
        t2    = ants.n4_bias_field_correction(t2)
        flair = ants.n4_bias_field_correction(flair)

        # MNI152 registration
        print("   -> Registering T1 -> MNI152 (Affine)...")
        mni           = ants.image_read(ants.get_ants_data("mni"))
        t1_reg        = ants.registration(fixed=mni, moving=t1, type_of_transform="Affine")
        t1_final      = t1_reg["warpedmovout"]
        t1_transforms = t1_reg["fwdtransforms"]

        print("   -> Registering T2 -> T1 -> MNI152...")
        t2_reg   = ants.registration(fixed=t1, moving=t2, type_of_transform="Rigid")
        t2_final = ants.apply_transforms(
            fixed=mni, moving=t2,
            transformlist=t1_transforms + t2_reg["fwdtransforms"]
        )

        print("   -> Registering FLAIR -> T1 -> MNI152...")
        fl_reg    = ants.registration(fixed=t1, moving=flair, type_of_transform="Rigid")
        fl_final  = ants.apply_transforms(
            fixed=mni, moving=flair,
            transformlist=t1_transforms + fl_reg["fwdtransforms"]
        )

        # Save
        print("   -> Saving...")
        ants.image_write(t1_final,  str(out_dir / f"{folder_name}_T1.nii.gz"))
        ants.image_write(t2_final,  str(out_dir / f"{folder_name}_T2.nii.gz"))
        ants.image_write(fl_final,  str(out_dir / f"{folder_name}_FLAIR.nii.gz"))
        generate_qc_snapshot(t1_final, t2_final, fl_final,
                             qc_dir / f"{folder_name}_QC.png")
        print(f"   -> Done: {folder_name}")

    except Exception as e:
        print(f"   [!] ERROR processing {folder_name}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir    = Path(__file__).resolve().parent
    data_root     = script_dir / "../../Data"
    raw_dir       = data_root / "raw"
    processed_dir = data_root / "processed"
    qc_dir        = processed_dir / "_QC_Snapshots"

    processed_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort all files in each modality folder
    t1_dir    = raw_dir / "T1W_synthstrip"
    t2_dir    = raw_dir / "T2W_synthstrip"
    flair_dir = raw_dir / "FLAIR_synthstrip"

    for d in (t1_dir, t2_dir, flair_dir):
        if not d.exists():
            print(f"ERROR: folder not found: {d}")
            return

    t1_files    = sorted(f for f in t1_dir.iterdir()    if f.is_file() and f.name.endswith(".nii.gz"))
    t2_files    = sorted(f for f in t2_dir.iterdir()    if f.is_file() and f.name.endswith(".nii.gz"))
    flair_files = sorted(f for f in flair_dir.iterdir() if f.is_file() and f.name.endswith(".nii.gz"))

    # Warn if folder sizes differ — extras in the longer folders are ignored
    counts = {"T1": len(t1_files), "T2": len(t2_files), "FLAIR": len(flair_files)}
    total  = min(counts.values())

    print(f"\nFiles found:  T1={counts['T1']}  T2={counts['T2']}  FLAIR={counts['FLAIR']}")
    if len(set(counts.values())) > 1:
        print(f"WARNING: folder sizes differ — processing {total} triplets (shortest folder wins).")
        print("         Check that all three folders have matching files.")
    print(f"Total triplets to process: {total}\n")

    skipped   = 0
    processed = 0
    errors    = 0

    # ── Core loop: one iteration = one matched triplet ────────────────────
    for i, (t1_path, t2_path, flair_path) in enumerate(
        zip(t1_files, t2_files, flair_files), start=1
    ):
        folder_name = derive_folder_name(t1_path, i)

        if is_already_processed(folder_name, processed_dir):
            print(f"  [{i:04d}/{total}] SKIP (already done): {folder_name}")
            skipped += 1
            continue

        print(f"  [{i:04d}/{total}] Processing: {folder_name}")
        try:
            process_case(folder_name, t1_path, t2_path, flair_path,
                         processed_dir, qc_dir)
            processed += 1
        except Exception as e:
            print(f"  [{i:04d}/{total}] ERROR: {folder_name}: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Finished.  Processed={processed}  Skipped={skipped}  Errors={errors}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
import re
import shutil
import ants
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Folder name from filename ─────────────────────────────────────────────────

def folder_name_from_path(path: Path) -> str:
    """
    Strip .nii.gz, _synthstrip, and the modality token from the filename.

    CALSNIC2_EDM_P110_T1w10_V1_run-02_synthstrip.nii.gz -> CALSNIC2_EDM_P110_V1_run-02
    CALSNIC2_CAL_C007_FLAIR_V1_synthstrip.nii.gz        -> CALSNIC2_CAL_C007_V1
    """
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    if name.endswith("_synthstrip"):
        name = name[: -len("_synthstrip")]
    name = re.sub(r'_(FLAIR\w*?|T[12]w\d*|T[12])(?=_|$)', '', name, flags=re.IGNORECASE)
    return name


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


# ── Per-case processing ───────────────────────────────────────────────────────

def process_case(folder_name: str,
                 t1_path: Path, t2_path: Path, flair_path: Path,
                 output_dir: Path, qc_dir: Path) -> None:
    out_dir = output_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{folder_name}] ===")
    print(f"   T1    : {t1_path.name}")
    print(f"   T2    : {t2_path.name}")
    print(f"   FLAIR : {flair_path.name}")

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

    # Register T1 -> MNI152
    print("   -> Registering T1 -> MNI152 (Affine)...")
    mni           = ants.image_read(ants.get_ants_data("mni"))
    t1_reg        = ants.registration(fixed=mni, moving=t1, type_of_transform="Affine")
    t1_final      = t1_reg["warpedmovout"]
    t1_transforms = t1_reg["fwdtransforms"]

    # Register T2 -> T1 -> MNI152
    print("   -> Registering T2 -> T1 -> MNI152...")
    t2_reg   = ants.registration(fixed=t1, moving=t2, type_of_transform="Rigid")
    t2_final = ants.apply_transforms(
        fixed=mni, moving=t2,
        transformlist=t1_transforms + t2_reg["fwdtransforms"]
    )

    # Register FLAIR -> T1 -> MNI152
    print("   -> Registering FLAIR -> T1 -> MNI152...")
    fl_reg   = ants.registration(fixed=t1, moving=flair, type_of_transform="Rigid")
    fl_final = ants.apply_transforms(
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir    = Path(__file__).resolve().parent
    data_root     = script_dir / "../../Data"
    raw_dir       = data_root / "raw"
    processed_dir = data_root / "processed"
    qc_dir        = processed_dir / "_QC_Snapshots"

    # Wipe and recreate the processed folder on every run
    if processed_dir.exists():
        print(f"Deleting existing processed folder: {processed_dir}")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)
    qc_dir.mkdir(parents=True)
    print("Processed folder cleared.\n")

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

    counts = {"T1": len(t1_files), "T2": len(t2_files), "FLAIR": len(flair_files)}
    total  = min(counts.values())

    print(f"Files found:  T1={counts['T1']}  T2={counts['T2']}  FLAIR={counts['FLAIR']}")
    if len(set(counts.values())) > 1:
        print(f"WARNING: folder sizes differ — processing {total} triplets (shortest folder wins).")
        print("         Verify that all three folders have the same files in the same order.")
    print(f"Total triplets to process: {total}\n")

    processed = 0
    errors    = 0

    for i, (t1_path, t2_path, flair_path) in enumerate(
        zip(t1_files, t2_files, flair_files), start=1
    ):
        folder_name = folder_name_from_path(t1_path)
        print(f"  [{i:04d}/{total}] {folder_name}")
        try:
            process_case(folder_name, t1_path, t2_path, flair_path,
                         processed_dir, qc_dir)
            processed += 1
        except Exception as e:
            print(f"  [{i:04d}/{total}] ERROR: {folder_name}: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Finished.  Processed={processed}  Errors={errors}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
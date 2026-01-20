import subprocess
import os
import re
import ants
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import platform

DOCKER_PLATFORM = "linux/amd64"



def parse_filename(filepath):
    """
    Extracts Subject ID, Visit ID, and Run number from filename.
    Returns (subject_id, visit_id, run_num) or None if parsing fails.
    """
    name = filepath.name

    
    match = re.search(r'_([CP]\d+)_', name)
    if not match:
        return None
    subject_id = match.group(1)
    
    match_visit = re.search(r'_(V\d+)', name)
    visit_id = match_visit.group(1) if match_visit else "V1" 
    
    match_run = re.search(r'_run-(\d+)', name)
    run_num = int(match_run.group(1)) if match_run else 1
    
    return (subject_id, visit_id, run_num)

def get_files_dict(raw_dir):
    """
    Scans raw directory and groups files by (Subject, Visit).
    Returns: dict[(Subject, Visit)] -> {'T1': (path, run), 'T2': (path, run), 'FLAIR': (path, run)}
    """
    modality_map = {
        'T1W': 'T1',
        'T2W': 'T2',
        'FLAIR': 'FLAIR'
    }
    
    subjects = {}
    
    for folder_name, mod_key in modality_map.items():
        search_path = raw_dir / folder_name
        if not search_path.exists():
            print(f"Warning: Directory {search_path} not found.")
            continue
            
        # Sort to ensure some determinism, though we rely on run numbers
        for filepath in sorted(search_path.glob("*.nii.gz")):
            parsed = parse_filename(filepath)
            if not parsed:
                continue
                
            subj, visit, run = parsed
            key = (subj, visit)
            
            if key not in subjects:
                subjects[key] = {'T1': None, 'T2': None, 'FLAIR': None}
            
            # If we have a previous file for this modality, check if this run is newer
            current = subjects[key][mod_key]
            if current is None or run > current[1]:
                subjects[key][mod_key] = (filepath, run)
                
    return subjects

def perform_n4_bias_correction(image):
    """
    Performs N4 Bias Field Correction using ANTs.
    """
    # N4 Bias Field Correction
    return ants.n4_bias_field_correction(image)

def reorient_to_ras(image):
    return ants.reorient_image2(image, orientation='RAS')

def run_docker_synthstrip(input_path, mask_path, host_folder):
    """Runs SynthStrip via Docker."""
    container_input = f"/data/{input_path.name}"
    container_mask = f"/data/{mask_path.name}"

    # Handle Windows paths for Docker volume mount
    host_folder_str = str(host_folder.resolve())
    if platform.system() == "Windows":
        host_folder_str = host_folder_str.replace('\\', '/')

    cmd = [
        "docker", "run", "--rm",
        "--platform", DOCKER_PLATFORM,
        "-v", f"{host_folder_str}:/data",
        "freesurfer/synthstrip",
        "-i", container_input,
        "-m", container_mask,
        "--no-csf"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def generate_qc_snapshot(t1, t2, flair, mask, filename):
    """Generates a QC image showing the middle slice of processed scans."""
    def get_mid_slice(img):
        if img is None: return np.zeros((100,100))
        arr = img.numpy()
        # Find center of mass or just middle index
        mid = arr.shape[2] // 2
        return np.rot90(arr[:, :, mid])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(get_mid_slice(t1), cmap="gray")
    axes[0].set_title("Processed T1")
    
    axes[1].imshow(get_mid_slice(t2), cmap="gray")
    axes[1].set_title("Processed T2")
    
    axes[2].imshow(get_mid_slice(flair), cmap="gray")
    axes[2].set_title("Processed FLAIR")
    
    for ax in axes: ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

def is_subject_processed(subj_id, visit_id, output_dir):
    """Checks if a subject has already been processed."""
    subj_folder_name = f"{subj_id}_{visit_id}"
    subj_out_dir = output_dir / subj_folder_name
    
    if not subj_out_dir.exists():
        return False
        
    expected_files = [
        subj_out_dir / f"{subj_id}_{visit_id}_T1.nii.gz",
        subj_out_dir / f"{subj_id}_{visit_id}_T2.nii.gz",
        subj_out_dir / f"{subj_id}_{visit_id}_FLAIR.nii.gz",
        subj_out_dir / f"{subj_id}_{visit_id}_mask.nii.gz"
    ]
    
    return all(f.exists() for f in expected_files)

def process_subject(subj_id, visit_id, paths, output_dir, qc_dir):
    t1_info = paths['T1']
    t2_info = paths['T2']
    flair_info = paths['FLAIR']
    
    if not (t1_info and t2_info and flair_info):
        print(f"Skipping {subj_id}_{visit_id}: Missing one or more modalities.")
        return

    t1_path, _ = t1_info
    t2_path, _ = t2_info
    flair_path, _ = flair_info
    
    subj_folder_name = f"{subj_id}_{visit_id}"
    subj_out_dir = output_dir / subj_folder_name
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Processing {subj_folder_name} ===")

    try:
        # Load
        print("   -> Loading...")
        t1 = ants.image_read(str(t1_path))
        t2 = ants.image_read(str(t2_path))
        flair = ants.image_read(str(flair_path))

        # Reorient first to standard RAS
        t1 = reorient_to_ras(t1)
        t2 = reorient_to_ras(t2)
        flair = reorient_to_ras(flair)
        
        # 1. N4 Bias Correction
        print("   -> Running N4 Bias Correction (ANTs)...")
        t1 = perform_n4_bias_correction(t1)
        t2 = perform_n4_bias_correction(t2)
        flair = perform_n4_bias_correction(flair)

        # 2. Registration to MNI 152
        print("   -> Loading MNI 152 Template...")
        mni = ants.image_read(ants.get_ants_data('mni'))

        print("   -> Registering T1 to MNI (Affine)...")
        # Register T1 to MNI
        t1_mni_reg = ants.registration(fixed=mni, moving=t1, type_of_transform='Affine')
        t1_final_unmasked = t1_mni_reg['warpedmovout']
        t1_transforms = t1_mni_reg['fwdtransforms']

        print("   -> Registering T2 to T1 and transforming to MNI...")
        # Register T2 to T1
        t2_t1_reg = ants.registration(fixed=t1, moving=t2, type_of_transform='Rigid')
        # Apply (T2 -> T1) then (T1 -> MNI)
        t2_final_unmasked = ants.apply_transforms(fixed=mni, moving=t2, 
                                                  transformlist=t1_transforms + t2_t1_reg['fwdtransforms'])

        print("   -> Registering FLAIR to T1 and transforming to MNI...")
        # Register FLAIR to T1
        flair_t1_reg = ants.registration(fixed=t1, moving=flair, type_of_transform='Rigid')
        # Apply (FLAIR -> T1) then (T1 -> MNI)
        flair_final_unmasked = ants.apply_transforms(fixed=mni, moving=flair, 
                                                     transformlist=t1_transforms + flair_t1_reg['fwdtransforms'])

        # 3. Skull Stripping (using Docker SynthStrip on MNI-registered FLAIR)
        temp_flair_path = subj_out_dir / "temp_flair_input.nii"
        temp_mask_path = subj_out_dir / "temp_mask.nii"
        
        ants.image_write(flair_final_unmasked, str(temp_flair_path))
        
        print("   -> Running SynthStrip...")
        run_docker_synthstrip(temp_flair_path, temp_mask_path, subj_out_dir)

        brain_mask = ants.image_read(str(temp_mask_path))

        # 4. Apply Mask
        print("   -> Applying Mask...")
        t1_final = ants.mask_image(t1_final_unmasked, brain_mask)
        t2_final = ants.mask_image(t2_final_unmasked, brain_mask)
        flair_final = ants.mask_image(flair_final_unmasked, brain_mask)

        # 5. Save
        print("   -> Saving outputs...")
        ants.image_write(t1_final, str(subj_out_dir / f"{subj_id}_{visit_id}_T1.nii.gz"))
        ants.image_write(t2_final, str(subj_out_dir / f"{subj_id}_{visit_id}_T2.nii.gz"))
        ants.image_write(flair_final, str(subj_out_dir / f"{subj_id}_{visit_id}_FLAIR.nii.gz"))
        ants.image_write(brain_mask, str(subj_out_dir / f"{subj_id}_{visit_id}_mask.nii.gz"))

        # QC
        generate_qc_snapshot(t1_final, t2_final, flair_final, brain_mask, qc_dir / f"{subj_id}_{visit_id}_QC.png")
        
        # Cleanup Temps
        if temp_flair_path.exists(): os.remove(temp_flair_path)
        if temp_mask_path.exists(): os.remove(temp_mask_path)
        
    except Exception as e:
        print(f"   [!] Error processing {subj_folder_name}: {e}")

def main():
    script_dir = Path(__file__).resolve().parent
    # Data is at ../../Data relative to src/preprossecing
    data_root = script_dir / "../../Data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    qc_dir = processed_dir / "_QC_Snapshots"
    

    
    # Recreate the main processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # QC directory is inside processed_dir
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for files in {raw_dir}...")
    subjects_dict = get_files_dict(raw_dir)
    
    if not subjects_dict:
        print("No matching subjects found. Check directory structure and filenames.")
        return

    print(f"Found {len(subjects_dict)} unique subject/visit sessions.")

    total_subjects_found = len(subjects_dict)
    current_processed_total = 0

    for (subj, visit), paths in subjects_dict.items():
        if is_subject_processed(subj, visit, processed_dir):
             print(f"Skipping {subj}_{visit} (Already Processed) ({current_processed_total + 1}/{total_subjects_found})")
             current_processed_total += 1
             continue

        print(f"Processing {subj}_{visit} ({current_processed_total + 1}/{total_subjects_found})...")
        process_subject(subj, visit, paths, processed_dir, qc_dir)
        
        current_processed_total += 1

if __name__ == "__main__":
    main()

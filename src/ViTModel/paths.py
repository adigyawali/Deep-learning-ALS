from pathlib import Path
import os

# Root for this module (`src/ViTModel`)
MODULE_DIR = Path(__file__).resolve().parent

# CNN features generated from full 3D T1/T2/FLAIR volumes.
# You can override this in Colab with env var: ALS_CNN_FEATURES_DIR
CNN_FEATURES_DIR = Path(
    os.environ.get("ALS_CNN_FEATURES_DIR", str((MODULE_DIR / "../cnn_features/cnn_multimodal/features").resolve()))
).resolve()

# ViT artifacts are kept separate from CNN artifacts.
# You can override this in Colab with env var: ALS_VIT_ARTIFACTS_DIR
ARTIFACTS_DIR = Path(
    os.environ.get("ALS_VIT_ARTIFACTS_DIR", str((MODULE_DIR / "../cnn_features/vit_multimodal").resolve()))
).resolve()
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

CHECKPOINT_PATH = CHECKPOINT_DIR / "vit_caf_best.pth"


def ensure_output_dirs() -> None:
    """Create all ViT output folders if they do not exist."""
    for directory in (ARTIFACTS_DIR, CHECKPOINT_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_runtime_paths(features_dir: str | None = None, artifacts_dir: str | None = None):
    """
    Build runtime paths, useful for Colab where paths are often in Google Drive.
    """
    runtime_features_dir = Path(features_dir).resolve() if features_dir else CNN_FEATURES_DIR
    runtime_artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir else ARTIFACTS_DIR
    runtime_checkpoint_dir = runtime_artifacts_dir / "checkpoints"
    runtime_metrics_dir = runtime_artifacts_dir / "metrics"
    runtime_checkpoint_path = runtime_checkpoint_dir / "vit_caf_best.pth"
    return runtime_features_dir, runtime_artifacts_dir, runtime_checkpoint_dir, runtime_metrics_dir, runtime_checkpoint_path

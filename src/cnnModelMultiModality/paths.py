from pathlib import Path

# Root of this module (`src/cnnModelMultiModality`)
MODULE_DIR = Path(__file__).resolve().parent

# Default processed MRI directory (outside `src`)
DATA_DIR = (MODULE_DIR / "../../Data/processed").resolve()

# Keep all CNN outputs grouped in one clean place
ARTIFACTS_DIR = (MODULE_DIR / "../cnn_features/cnn_multimodal").resolve()
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
FEATURES_DIR = ARTIFACTS_DIR / "features"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

CHECKPOINT_PATH = CHECKPOINT_DIR / "encoder_weights.pth"


def ensure_output_dirs() -> None:
    """Create all artifact folders used by training/evaluation/extraction."""
    for directory in (ARTIFACTS_DIR, CHECKPOINT_DIR, FEATURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

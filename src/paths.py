from pathlib import Path


SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


if __name__ == "__main__":
    all_dirs = [
        SRC_DIR,
        ROOT_DIR,
        DATA_DIR
    ]
    for d in all_dirs:
        d.mkdir(parents=True, exist_ok=True)

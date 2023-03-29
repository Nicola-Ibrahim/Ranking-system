from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_SPACES_DATA_PATH = BASE_DIR / "data/raw/spaces_dummy_data.json"
PRE_PROC_SPACES_TIME_PATH = BASE_DIR / "data/interim/spaces_encoded_data.json"
PROC_SPACES_DATA_PATH = BASE_DIR / "data/processed/spaces_processed_data.json"
DECISION_MATRIX_PATH = BASE_DIR / "data/processed/decision_matrix.pkl"
COMB_SCORES_PATH = BASE_DIR / "data/results/scores.csv"

# Set maximum retrieved spaces from database
MAX_SEARCH_RANGE = 10

from pathlib import Path
PROJECT_ROOT =Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT/"data"/"raw"
PROCESSED_DIR = PROJECT_ROOT/"data"/"processed"

# Column R is at index 17 starting from 0 Thus, we take the waveform from col R onwards.
WAVEFORM_START_COL_INDEX=17

RANDOM_SEED=42
TEST_SIZE=0.2
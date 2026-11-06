from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATA_FILE = DATA_DIR / "College_student_behaviourial_data.csv"
TARGET = "risk_level"

# Exclude identifiers, target, and variables that may cause target leakage.
DROP_FROM_FEATURES = [
    "student_id",
    "GPA",
    "avg_course_grade",
    "enrollment_status",
    TARGET,
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

for folder in [DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

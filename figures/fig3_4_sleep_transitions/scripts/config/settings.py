import os

# Base data directory
# Can be set via environment variable FIG3_BASE_DIR
BASE_DIR = os.environ.get(
    "FIG3_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
)

# Output directory
RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS")
os.makedirs(RESULTS_DIR, exist_ok=True)

# fMRI timing

WIN_SECONDS = 60

# Transition patterns (pre → post)
TRANSITION_PATTERNS = [
    (1, 3),  # wake → NREM
    (3, 1),  # NREM → wake
]

# Baseline and analysis windows (in seconds)
BASELINE_WINDOW = (-10, 0)   # usata per stats & cluster
STRICT_POST_WINDOW = (0, 10)  # usata per strict stats

DETAILED_WINDOWS = {
    "baseline": (-10, 0),
    "early": (0, 5),
    "mid": (5, 10),
    "late": (10, 30),
}

# Cluster-based permutation analysis window
CLUSTER_ANALYSIS_WINDOW = (0, 30)
N_PERMUTATIONS = 5000
ALPHA_CLUSTER = 0.05

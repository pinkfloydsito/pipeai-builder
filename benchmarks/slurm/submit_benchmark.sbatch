#!/bin/bash
#SBATCH --job-name=automl-benchmark
#SBATCH --array=0-9                    # 10 datasets (0-indexed)
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --time=02:00:00                # 2 hours per dataset
#SBATCH --mem=16G                      # 16GB RAM
#SBATCH --cpus-per-task=4              # 4 CPUs for TPOT
#SBATCH --partition=compute            # Adjust to your cluster

# ============================================================================
# AutoML vs LLM Benchmark - SLURM Array Job
# Runs benchmark on 10 datasets in parallel
# ============================================================================

# Exit on error
set -e

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8  # if using GPU

# Configuration - EDIT THESE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="${PROJECT_ROOT}/benchmarks/slurm/results"
TPOT_TIME_MINS=10
LLM_PROVIDER="deepseek"  # or "openai"

# API key from environment or file
if [ -z "$LLM_API_KEY" ]; then
    if [ -f "${SCRIPT_DIR}/.api_key" ]; then
        LLM_API_KEY=$(cat "${SCRIPT_DIR}/.api_key")
    else
        echo "ERROR: LLM_API_KEY not set and .api_key file not found"
        exit 1
    fi
fi

# Dataset IDs (must match array size)
DATASETS=(61 37 44 1462 1464 1510 40984 40975 40966 40982)

# Get dataset ID for this array task
DATASET_ID=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset ID: $DATASET_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR/pipelines"
mkdir -p "$SCRIPT_DIR/logs"

# Activate virtual environment (adjust path)
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Run benchmark for this dataset
python "${SCRIPT_DIR}/run_single_dataset.py" \
    --dataset-id "$DATASET_ID" \
    --api-key "$LLM_API_KEY" \
    --tpot-time "$TPOT_TIME_MINS" \
    --output-dir "$OUTPUT_DIR" \
    --provider "$LLM_PROVIDER"

echo "============================================================"
echo "End time: $(date)"
echo "============================================================"

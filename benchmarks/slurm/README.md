# SLURM Benchmark Jobs

Run the AutoML (TPOT) vs LLM Designer benchmark on 10 datasets using SLURM array jobs.

## Quick Start

```bash
# 1. Set your API key
export LLM_API_KEY="your-api-key"
# Or create a .api_key file:
echo "your-api-key" > .api_key

# 2. Submit the job array
sbatch submit_benchmark.sbatch

# 3. Monitor jobs
squeue -u $USER

# 4. After completion, merge results
python merge_results.py
```

## Files

| File | Description |
|------|-------------|
| `submit_benchmark.sbatch` | Main SLURM submission script (array job) |
| `run_single_dataset.py` | Python script to run benchmark on one dataset |
| `merge_results.py` | Merge results after all jobs complete |
| `datasets.txt` | List of 10 OpenML dataset IDs |

## Datasets

| ID | Name | Samples | Features | Classes |
|----|------|---------|----------|---------|
| 61 | iris | 150 | 4 | 3 |
| 37 | diabetes | 768 | 8 | 2 |
| 44 | spambase | 4601 | 57 | 2 |
| 1462 | banknote-authentication | 1372 | 4 | 2 |
| 1464 | blood-transfusion | 748 | 4 | 2 |
| 1510 | wdbc (breast cancer) | 569 | 30 | 2 |
| 40984 | segment | 2310 | 19 | 7 |
| 40975 | car | 1728 | 6 | 4 |
| 40966 | MiceProtein | 1080 | 77 | 8 |
| 40982 | steel-plates-fault | 1941 | 27 | 7 |

## Configuration

Edit `submit_benchmark.sbatch` to adjust:

```bash
#SBATCH --time=02:00:00      # Time per dataset
#SBATCH --mem=16G            # Memory
#SBATCH --cpus-per-task=4    # CPUs for TPOT
#SBATCH --partition=compute  # Your cluster partition

TPOT_TIME_MINS=10            # TPOT optimization time
LLM_PROVIDER="deepseek"      # or "openai"
```

## Output Structure

```
slurm/
├── logs/
│   ├── benchmark_12345_0.out   # Job logs
│   └── benchmark_12345_0.err
└── results/
    ├── results_dataset_61.csv
    ├── results_dataset_37.csv
    ├── ...
    ├── pipelines/
    │   ├── iris_baseline_RandomForestClassifier.json
    │   ├── iris_tpot_*.json
    │   └── iris_llm_*.json
    ├── benchmark_results_all.csv   # Merged (after merge_results.py)
    └── benchmark_summary.json      # Summary stats
```

## Local Testing

Test on a single dataset before submitting to SLURM:

```bash
python run_single_dataset.py \
    --dataset-id 61 \
    --api-key YOUR_KEY \
    --tpot-time 2 \
    --output-dir test_results
```

## Troubleshooting

**Job fails immediately:**
- Check logs in `logs/benchmark_*.err`
- Verify API key is set correctly
- Ensure Python environment has all dependencies

**Out of memory:**
- Increase `--mem` in sbatch script
- Reduce TPOT population size

**Timeout:**
- Increase `--time` limit
- Reduce `TPOT_TIME_MINS`

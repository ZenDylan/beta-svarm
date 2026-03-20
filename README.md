# β-SVARM: Beta-weighted Semivalue Data Valuation via Stratified Non-Marginal Decomposition

## Quick Start

### 1. Clone
git clone https://github.com/<YOUR_USERNAME>/beta-svarm.git
cd beta-svarm

### 2. Create conda environment
conda env create -f environment.yml
conda activate beta-svarm

### 3. Run verification test (30 seconds)
python tests/test_toy.py

### 4. Run all Claim 1 experiments (estimated 1-3 hours)
python experiments/run_claim1.py

### 5. Check results
Results saved in results/ directory as PNG files.

## Experiment Outputs
- results/convergence_Gaussian.png
- results/convergence_Adult.png
- results/convergence_MNIST.png
- results/runtime_Adult.png
- results/multisemivalue_Adult.png

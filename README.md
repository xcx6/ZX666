# SGX-FL: Secure Federated Learning with Malicious Client Detection

This is a federated learning framework with malicious client detection capabilities, extracted from the oldXFL project.

## Project Structure

```
SGX-FL/
├── run_independent_detector_test.sh    # Main entry script
├── test_independent_detectors_training.py  # Training script with detection
├── independent_detectors_test.py        # Independent detector tester (direction similarity)
├── Algorithm/                          # Training algorithms
│   └── Training_XFL_SmallData.py       # XFL training with small data
├── models/                             # Model definitions
├── utils/                              # Utility functions
├── attacks/                             # Attack implementations
├── getAPOZ.py                          # APOZ calculation utilities
├── data_collector.py                   # Data collection utilities
├── wandbUtils.py                       # WandB utilities (optional)
└── requirements.txt                    # Python dependencies
```

**Note**: The current detection system uses `direction_similarity` detector (implemented in `independent_detectors_test.py`), which calculates cosine similarity between update directions from external and TEE models.

## Quick Start

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x run_independent_detector_test.sh
```

### Usage

Run the default configuration (ResNet18 + CIFAR-10):
```bash
./run_independent_detector_test.sh
```

Run with different models and datasets:
```bash
# LeNet5 + MNIST
MODEL=lenet5 DATASET=mnist ./run_independent_detector_test.sh

# ResNet20 + Fashion-MNIST
MODEL=resnet20 DATASET=fmnist ./run_independent_detector_test.sh
```

Run with different attack types:
```bash
# Label flipping attack
ATTACK_TYPE=label_flipping ./run_independent_detector_test.sh

# Noise injection attack
ATTACK_TYPE=noise_injection ./run_independent_detector_test.sh

# No attack
ATTACK_TYPE=no_attack ./run_independent_detector_test.sh
```

Run with different data distributions:
```bash
# IID data
DATA_DISTRIBUTION=iid ./run_independent_detector_test.sh

# Non-IID data (default)
DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh

# Non-IID with different severity levels
NONIID_CASE=1 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh  # Mild
NONIID_CASE=2 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh  # Moderate (default)
NONIID_CASE=3 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh  # Extreme
```

## Configuration

### Environment Variables

#### Model and Dataset Configuration
- `MODEL`: Model type
  - Default: `resnet`
  - Options: `resnet`, `resnet20`, `lenet5`, `vgg`
- `DATASET`: Dataset name
  - Default: `cifar10`
  - Options: `cifar10`, `mnist`, `fmnist`
  - Automatically adjusts hyperparameters based on dataset:
    - **MNIST**: `NUM_USERS=100`, `LOCAL_EP=10`, `BS=32`, `LR=0.01`
    - **Fashion-MNIST**: `NUM_USERS=100`, `LOCAL_EP=15`, `BS=32`, `LR=0.01`
    - **CIFAR-10**: `NUM_USERS=100`, `LOCAL_EP=20`, `BS=32`, `LR=0.01`
    - **Note**: Batch size is set to 32 for all datasets (uses default value from `utils/options.py`)

#### Attack Configuration
- `ATTACK_TYPE`: Attack scenario
  - Default: `label_flipping`
  - Options:
    - `label_flipping`: Label flipping attack (100% flip rate, all labels randomly flipped)
    - `noise_injection`: Noise injection attack (100% of data with Gaussian noise, std=0.25)
    - `no_attack`: No attack mode
- `ENABLE_DEFENSE`: Defense mode
  - Default: `1` (enabled)
  - `1`: Defense mode (detector controls aggregation, rejects malicious clients)
  - `0`: Observation mode (detector only records data, does not affect aggregation)

#### Data Distribution Configuration
- `DATA_DISTRIBUTION`: Data distribution type
  - Default: `noniid`
  - Options: `iid`, `noniid`
- `NONIID_CASE`: Non-IID severity level (only effective when `DATA_DISTRIBUTION=noniid`)
  - Default: `2` (moderate)
  - Options:
    - `1`: Mild heterogeneity → Auto-maps to: `ACTUAL_CASE=4`, `DATA_BETA=0.8`, `PROX_ALPHA=0.01`
    - `2`: Moderate heterogeneity → Auto-maps to: `ACTUAL_CASE=5`, `DATA_BETA=0.5`, `PROX_ALPHA=0.1`
    - `3`: Extreme heterogeneity → Auto-maps to: `ACTUAL_CASE=6`, `DATA_BETA=0.1`, `PROX_ALPHA=0.5`
  - Note: `ACTUAL_CASE >= 4` uses Dirichlet distribution for data splitting
- `DATA_BETA`: Dirichlet distribution parameter α (optional override)
  - If not set, automatically mapped from `NONIID_CASE`
  - Controls data heterogeneity: smaller α = more heterogeneous

#### Aggregation Configuration
- `USE_FEDPROX`: Use FedProx aggregation
  - Default: `1` (enabled)
  - `1`: FedProx (adds proximal term: `loss += (μ/2)||w - w_global||²`)
  - `0`: FedAvg (simple average, no regularization)
  - Recommended: Use FedProx for Non-IID scenarios
- `PROX_ALPHA`: FedProx regularization strength μ (optional override)
  - If not set, automatically mapped from `NONIID_CASE`
  - Controls regularization strength: larger μ = stronger constraint

#### Training Configuration
- `EPOCHS`: Number of training rounds
  - Default: `50`
- `RANDOM_SEED`: Random seed for reproducibility
  - Default: Not set (uses random seed, different results each run)
  - Set to a specific number for reproducible results

#### Detection Configuration
- Detection thresholds (internal, not configurable via environment variables):
  - Label flipping: `direction_similarity < 0.1` → reject
  - Noise injection: `direction_similarity < 0.24` → reject
  - Warm-up rounds: First 3 rounds (0-2) skip detection (cold start period)

### Hyperparameters Summary

Based on dataset selection, the following hyperparameters are automatically configured:

| Dataset | Clients | Local Epochs | Batch Size | Learning Rate | Warm-up Rounds |
|---------|---------|--------------|------------|---------------|----------------|
| MNIST | 100 | 10 | 32 | 0.01 | 3 |
| Fashion-MNIST | 100 | 15 | 32 | 0.01 | 3 |
| CIFAR-10 | 100 | 20 | 32 | 0.01 | 3 |

**Client Selection**:
- Warm-up period (rounds 0-2): Fixed 10 benign clients
- Normal period (round 3+): 20 clients (10 benign + 10 malicious, if attack enabled)

**Non-IID Parameter Mapping**:
- `NONIID_CASE=1` → `ACTUAL_CASE=4`, `DATA_BETA=0.8`, `PROX_ALPHA=0.01` (mild)
- `NONIID_CASE=2` → `ACTUAL_CASE=5`, `DATA_BETA=0.5`, `PROX_ALPHA=0.1` (moderate, default)
- `NONIID_CASE=3` → `ACTUAL_CASE=6`, `DATA_BETA=0.1`, `PROX_ALPHA=0.5` (extreme)

## Features

- **Malicious Client Detection**: Uses direction similarity detector (cosine similarity of update directions) to identify malicious clients
- **Flexible Attack Scenarios**: Supports label flipping, noise injection, and no-attack modes
- **Data Distribution Options**: Supports both IID and Non-IID data distributions with configurable heterogeneity levels
- **Multiple Models**: Supports ResNet, LeNet5, VGG, and other models
- **Multiple Datasets**: Supports CIFAR-10, MNIST, Fashion-MNIST with automatic hyperparameter adjustment
- **Adaptive Aggregation**: FedProx for Non-IID scenarios with automatic regularization strength mapping

## Detection Mechanism

The system uses **direction similarity** detector:
- **Method**: Calculates cosine similarity between update directions from external model (potentially malicious) and TEE model (trusted)
- **Threshold**: 
  - Label flipping attack: `direction_similarity < 0.1` → reject
  - Noise injection attack: `direction_similarity < 0.24` → reject
- **Warm-up**: First 3 rounds skip detection to establish baseline
- **Decision**: If similarity below threshold → client is rejected from aggregation

## Output

Training results are saved as JSON files with the naming pattern:
```
independent_test_{MODEL}_{ATTACK_TYPE}_{DISTRIBUTION}_{timestamp}.json
```

The output includes:
- Detection results for each client in each round
- Global model accuracy and loss per round
- Complete training process records
- Final statistics (accuracy, precision, recall, etc.)

## Notes

- The script automatically handles warm-up rounds before enabling detection
- Detection thresholds are optimized for different attack types
- The framework uses FedProx aggregation for Non-IID scenarios by default
- Non-IID parameters (`DATA_BETA`, `PROX_ALPHA`) are automatically mapped from `NONIID_CASE` if not manually set
- All hyperparameters are dynamically configured based on dataset selection


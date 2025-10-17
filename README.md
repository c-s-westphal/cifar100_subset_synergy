# VGG CIFAR-100 First-Layer Masking and Mutual Information

This repository trains VGG models with Global Average Pooling (GAP) on CIFAR-100 and measures the effect of first-layer masking on mutual information. It focuses on a single configuration (BN=yes, Aug=yes, Dropout=yes, Optimizer=Adam) across 5 architectures and 3 random seeds.

## Overview

This is a simplified study focused on measuring how masking random subsets of first-layer activations affects mutual information between model predictions and true labels. Unlike ablation studies, this repository uses a single optimized configuration for all experiments.

## Repository Structure

```
cifar100_subset_synergy/
├── models/
│   └── vgg_standard.py          # VGG architectures (9, 11, 13, 16, 19) with GAP
├── scripts/
│   ├── jobs_vgg9.txt            # Job file for VGG9 (3 seeds)
│   ├── jobs_vgg11.txt           # Job file for VGG11 (3 seeds)
│   ├── jobs_vgg13.txt           # Job file for VGG13 (3 seeds)
│   ├── jobs_vgg16.txt           # Job file for VGG16 (3 seeds)
│   ├── jobs_vgg19.txt           # Job file for VGG19 (3 seeds)
│   └── job_manager.py           # SLURM job submission manager
├── visualizations/              # Plotting scripts (to be added)
├── results/                     # Training results (npz, json)
├── plots/                       # Generated plots
├── checkpoints/                 # Model checkpoints
├── data/                        # CIFAR-100 data
├── train_vgg.py                 # Main training script
└── README.md                    # This file
```

## Model Architecture

### VGG with Global Average Pooling

All VGG models use Global Average Pooling instead of traditional fully-connected classifier heads:

```python
class VGG(nn.Module):
    def __init__(self, arch: str, num_classes: int = 100,
                 use_batchnorm: bool = True, use_dropout: bool = True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[arch], use_batchnorm)

        # Global Average Pooling + Dropout + Linear classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 512, H, W) -> (B, 512, 1, 1)

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)      # Conv features
        x = self.pool(x)          # (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # (B, 512)
        x = self.classifier(x)    # (B, num_classes)
        return x
```

### Architectures

- **VGG9**: 9 convolutional layers
- **VGG11**: 11 convolutional layers
- **VGG13**: 13 convolutional layers
- **VGG16**: 16 convolutional layers
- **VGG19**: 19 convolutional layers

## Configuration

### Single Configuration (No Ablations)

- **Dataset**: CIFAR-100
- **Batch Normalization**: Yes
- **Data Augmentation**: Yes (RandAugment, CutMix, RandomCrop, RandomHorizontalFlip)
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0
- **Dropout**: Yes (0.5 dropout rate between GAP and final linear layer)
- **Batch Size**: 256
- **Epochs**: 300
- **Seeds**: 0, 1, 2 (3 runs per architecture)

### Learning Rate Schedule

- **Warmup**: 10 epochs (linear increase from 0.1% to 100% of LR)
- **Cosine Annealing**: Remaining epochs (decay to 1e-5)

### Data Augmentation

```python
# Training augmentation
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# CutMix applied during training (alpha=1.0)
# Label smoothing (0.1)
```

## First-Layer Masking Mechanism

The masking mechanism randomly selects subsets of activation elements in the first convolutional layer output to measure information redundancy.

### Mask Generation

```python
def generate_random_channel_masks(
    n_channels: int,
    height: int,
    width: int,
    n_masks: int,
    seed: int = 42
) -> List[np.ndarray]:
    """Generate random masks for element selection.

    Each mask randomly selects between 1 and (n_channels*h*w - 2)
    individual elements to keep.

    Returns masks where True = keep element, False = zero out element.
    """
    # Randomly select subset_size elements to keep
    # Shape: (n_channels, height, width)
```

### Masking Hook

```python
class ChannelMaskingHook:
    """Hook to mask individual elements in convolutional layer output."""
    def __init__(self, mask: np.ndarray):
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """Apply element-wise masking during forward pass."""
        masked_output = output.clone()
        masked_output = masked_output * self.mask.unsqueeze(0).to(output.device)
        return masked_output
```

## Mutual Information Evaluation

### MI Calculation

```python
from sklearn.metrics import mutual_info_score

def calculate_mutual_information(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate discrete mutual information: I(Y; predictions)

    Maximum MI is log2(100) ≈ 6.64 bits for CIFAR-100.
    """
    return mutual_info_score(labels, predictions)
```

### MI Evaluation Protocol

1. **Full Model MI**: Calculate MI between predictions and labels with no masking
2. **Masked MI**: Calculate MI for N random masks (subsets of first-layer activations)
3. **MI Difference**: `MI_full - mean(MI_masked)` measures information loss from masking

### Evaluation Schedule

- **During Training**: Every 10 epochs
  - 20 random masks
  - 20 batches maximum
- **Final Evaluation**: After training
  - 40 random masks
  - 40 batches maximum

## Usage

### Training a Single Model

```bash
python train_vgg.py \
  --arch vgg11 \
  --seed 0 \
  --epochs 300 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 0.0 \
  --eval_interval 10 \
  --n_masks_train 20 \
  --n_masks_final 40 \
  --max_eval_batches_train 20 \
  --max_eval_batches_final 40
```

### Generating SLURM Job Scripts

```bash
# Generate job scripts for all architectures
python scripts/job_manager.py

# Or generate without submitting
python scripts/job_manager.py --generate_only
```

This creates:
- `scripts/job_manager_vgg9.sh`
- `scripts/job_manager_vgg11.sh`
- `scripts/job_manager_vgg13.sh`
- `scripts/job_manager_vgg16.sh`
- `scripts/job_manager_vgg19.sh`

### Submitting Jobs

```bash
# Submit individual architecture
qsub scripts/job_manager_vgg11.sh

# Submit all architectures
for script in scripts/job_manager_*.sh; do qsub $script; done
```

### Job Management

Jobs are configured to:
- Skip training if checkpoint and results already exist
- Use 16GB memory
- Request 24-hour runtime
- Use GPU resources

## Results Format

### NPZ Format

Each training run saves results in `.npz` format:

```python
{
    'epochs_evaluated': array([1, 10, 20, ..., 300]),
    'mi_history': array([...]),           # MI difference at each eval
    'train_acc_aug_history': array([...]), # Training acc (augmented)
    'train_acc_clean_history': array([...]), # Training acc (clean)
    'test_acc_history': array([...]),
    'gen_gap_history': array([...]),
    'final_mi_full': float,               # Final MI (full model)
    'final_mean_mi_masked': float,        # Final MI (mean over masks)
    'final_mi_diff': float,               # Final MI difference
    'final_train_acc_aug': float,
    'final_train_acc_clean': float,
    'final_test_acc': float,
    'final_gen_gap': float,
    'arch': str,
    'seed': int,
    'use_batchnorm': True,
    'use_augmentation': True,
    'use_dropout': True,
    'optimizer': 'adam'
}
```

### JSON Format

Results are also saved in human-readable JSON format with the same information.

## Expected Results

### Total Jobs

- 5 architectures × 3 seeds = **15 total jobs**

### File Organization

```
checkpoints/
├── vgg9_seed0_final.pt
├── vgg9_seed1_final.pt
├── vgg9_seed2_final.pt
├── vgg11_seed0_final.pt
...
└── vgg19_seed2_final.pt

results/
├── vgg9_seed0_results.npz
├── vgg9_seed0_results.json
├── vgg9_seed1_results.npz
├── vgg9_seed1_results.json
...
└── vgg19_seed2_results.json
```

## Key Differences from CIFAR-10 Repo (comparing_bn_adam)

1. **No Ablations**: Single configuration only (BN=yes, Aug=yes, Dropout=yes, Adam)
2. **Dataset**: CIFAR-100 instead of CIFAR-10 (100 classes vs 10 classes)
3. **Architectures**: Added VGG9, includes VGG9/11/13/16/19
4. **Focus**: Measuring masking effects on MI across architectures and seeds
5. **Model**: Uses Global Average Pooling (GAP) instead of FC classifier head

## Verification

After training completes, verify:

1. **Checkpoints exist**: 15 checkpoint files (5 architectures × 3 seeds)
2. **Results exist**: 30 result files (15 .npz + 15 .json)
3. **MI evaluation**: Each result includes MI measurements every 10 epochs + final comprehensive evaluation
4. **Identical MI/masking logic**: Same implementation as CIFAR-10 repo

## Dependencies

```
torch
torchvision
numpy
scikit-learn
```

## References

- VGG Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
- CIFAR-100 Dataset: 100 classes with 600 images each (500 training, 100 test)
- Related Repository: `comparing_bn_adam` (CIFAR-10 ablation study)

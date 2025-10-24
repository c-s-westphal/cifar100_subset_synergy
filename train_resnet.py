"""
Training script for ResNet models on CIFAR-10 with integrated MI evaluation.

Trains ResNet models (20, 32, 44, 56, 110) and evaluates
the effect of masking after first skip connection on mutual information.
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import mutual_info_score

from models.resnet_standard import ResNet20, ResNet32, ResNet44, ResNet56, ResNet74, ResNet110


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation.

    Args:
        x: Input tensor (batch_size, C, H, W)
        y: Target labels (batch_size,)
        alpha: CutMix parameter (default: 1.0)

    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    indices = torch.randperm(x.size(0))
    shuffled_y = y[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y, shuffled_y, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix.

    Returns:
        Tuple of (bbx1, bby1, bbx2, bby2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch_size, num_classes)
            target: Targets (batch_size,)
        """
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class

        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1).mean()

        return loss


def get_data_loaders(
    batch_size: int = 256,
    num_workers: int = 4,
    data_root: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders with standard augmentation.

    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of data loading workers
        data_root: Root directory for CIFAR-10 data
    """
    # CIFAR-10 normalization
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Train transform (standard augmentation: RandomCrop + RandomHorizontalFlip)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def get_eval_loader(
    batch_size: int = 256,
    num_workers: int = 4,
    data_root: str = './data'
) -> DataLoader:
    """Create evaluation data loader (CIFAR-10, no augmentation, no shuffling)."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    eval_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=eval_transform
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return eval_loader


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0
) -> float:
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    batches_processed = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batches_processed += 1
            if max_batches and batches_processed >= max_batches:
                break

    return 100. * correct / total


def generate_random_channel_masks(
    n_channels: int,
    height: int,
    width: int,
    n_masks: int,
    seed: int = 42
) -> List[np.ndarray]:
    """Generate random masks for element selection in first conv layer activations.

    Each mask randomly selects between 1 and (n_channels*h*w - 2) individual elements to keep.
    Returns masks where True = keep element, False = zero out element.
    Mask shape: (n_channels, height, width)
    """
    np.random.seed(seed)
    masks = []

    # Total number of elements in activation tensor
    total_elements = n_channels * height * width

    # Maximum subset size: all elements except 2
    max_subset_size = max(1, total_elements - 2)

    for _ in range(n_masks):
        # Random subset size between 1 and max_subset_size
        subset_size = np.random.randint(1, max_subset_size + 1)

        # Create flat mask and randomly select elements to KEEP
        flat_mask = np.zeros(total_elements, dtype=bool)
        selected_indices = np.random.choice(total_elements, subset_size, replace=False)
        flat_mask[selected_indices] = True

        # Reshape to 3D: (n_channels, height, width)
        mask = flat_mask.reshape(n_channels, height, width)
        masks.append(mask)

    return masks


class ChannelMaskingHook:
    """Hook to mask individual elements in convolutional layer output."""
    def __init__(self, mask: np.ndarray):
        """
        Args:
            mask: Boolean array where True = keep element, False = zero out element
                  Shape: (channels, height, width) for 3D element masking
        """
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """Apply element-wise masking during forward pass."""
        # output shape: (batch, channels, height, width)
        # mask shape: (channels, height, width)
        masked_output = output.clone()

        # Apply mask element-wise across all spatial positions and channels
        # Broadcast mask across batch dimension
        masked_output = masked_output * self.mask.unsqueeze(0).to(output.device)

        return masked_output


def get_first_conv_block_output(model: nn.Module) -> nn.Module:
    """Get the output after the first skip connection.

    Returns the first residual block (layer1[0]) which contains the first skip connection.
    """
    # ResNet architecture: mask after first skip connection in layer1[0]
    if hasattr(model, 'layer1'):
        return model.layer1[0]

    # VGG architecture: features sequential module
    if hasattr(model, 'features'):
        # Find the first Conv2d, then find the ReLU that follows it
        found_first_conv = False

        for module in model.features:
            if isinstance(module, nn.Conv2d) and not found_first_conv:
                found_first_conv = True
            elif found_first_conv and isinstance(module, nn.ReLU):
                # This is the ReLU after Conv (and possibly BN)
                return module

        if found_first_conv:
            raise ValueError("Found first conv but no ReLU after it")

    raise ValueError("Could not find first conv block in model")


def get_predictions_and_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mask: np.ndarray = None,
    hook_layer: nn.Module = None,
    max_batches: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and true labels.

    Args:
        model: The model to evaluate
        data_loader: Data loader
        device: Device to use
        mask: Optional mask to apply at hook_layer
        hook_layer: Optional layer to hook for masking
        max_batches: Maximum number of batches to process (0 = all)

    Returns:
        (predictions, labels) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_labels = []

    hook_handle = None
    if mask is not None and hook_layer is not None:
        hook = ChannelMaskingHook(mask)
        hook_handle = hook_layer.register_forward_hook(hook)

    try:
        batches_processed = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

                batches_processed += 1
                if max_batches and batches_processed >= max_batches:
                    break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    return predictions, labels


def calculate_mutual_information(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate mutual information between predictions and true labels.

    Uses discrete mutual information: I(Y; predictions)
    Maximum MI is log2(10) ≈ 3.32 bits for CIFAR-10.
    """
    return mutual_info_score(labels, predictions)


def evaluate_first_layer_mi(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    n_subsets: int,
    seed: int = 42,
    max_batches: int = 0
) -> Tuple[float, float, float]:
    """Evaluate MI difference between full and masked activations after first skip connection.

    Returns:
        (mi_full, mean_mi_masked, mi_difference)
    """
    model.eval()

    # Get the output after first skip connection (layer1[0])
    first_block_output = get_first_conv_block_output(model)

    # Determine first layer dimensions
    sample_batch = next(iter(eval_loader))
    sample_input = sample_batch[0][:1].to(device)

    hook_output = None
    def capture_hook(module, input, output):
        nonlocal hook_output
        hook_output = output

    handle = first_block_output.register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(sample_input)
    handle.remove()

    n_channels = hook_output.shape[1]
    height = hook_output.shape[2]
    width = hook_output.shape[3]

    # Generate element masks
    masks = generate_random_channel_masks(n_channels, height, width, n_subsets, seed)

    # Get predictions for full model
    full_predictions, labels = get_predictions_and_labels(
        model, eval_loader, device, mask=None, hook_layer=None, max_batches=max_batches
    )

    # Calculate MI for full model
    mi_full = calculate_mutual_information(full_predictions, labels)

    # Calculate MI for each masked version
    masked_mis = []
    for mask in masks:
        masked_predictions, _ = get_predictions_and_labels(
            model, eval_loader, device, mask=mask, hook_layer=first_block_output, max_batches=max_batches
        )
        mi_masked = calculate_mutual_information(masked_predictions, labels)
        masked_mis.append(mi_masked)

    # Calculate mean MI across all masks
    mean_mi_masked = np.mean(masked_mis)
    mi_difference = mi_full - mean_mi_masked

    return mi_full, mean_mi_masked, mi_difference


def separate_parameters_for_weight_decay(model: nn.Module):
    """Separate parameters into groups for selective weight decay.

    Apply weight decay only to conv/linear weights, not to BatchNorm params or any biases.

    Returns:
        List of parameter dicts for optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay on biases or BatchNorm parameters
        if 'bias' in name or 'bn' in name or isinstance(param, nn.BatchNorm2d):
            no_decay_params.append(param)
        else:
            # Apply weight decay to conv/linear weights
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': None},  # Will be set by optimizer
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cutmix_alpha: float = 1.0,
    grad_clip: float = 0.0
) -> Tuple[float, float]:
    """Train for one epoch with optional CutMix.

    If cutmix_alpha=0.0, CutMix is disabled and standard training is used.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if cutmix_alpha > 0:
            # Apply CutMix augmentation
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            outputs = model(inputs)

            # Mixed loss for CutMix
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

            # Mixed accuracy for CutMix
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            # Standard training without CutMix
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Standard accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10 with MI evaluation')

    # Model arguments
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet74', 'resnet110'],
                        help='Model architecture')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=600,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (base LR)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (applied to all parameters with AdamW)')
    parser.add_argument('--target_train_acc', type=float, default=99.0,
                        help='Target clean train accuracy for early stopping')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate MI every N epochs')
    parser.add_argument('--n_masks_train', type=int, default=20,
                        help='Number of masks for training MI evaluation')
    parser.add_argument('--n_masks_final', type=int, default=80,
                        help='Number of masks for final MI evaluation')
    parser.add_argument('--max_eval_batches_train', type=int, default=20,
                        help='Max batches for training MI evaluation')
    parser.add_argument('--max_eval_batches_final', type=int, default=80,
                        help='Max batches for final MI evaluation')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available! "
                "This job should be rescheduled to a working GPU node. "
                "If you see this error, the SLURM node has GPU issues."
            )
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Create model (single configuration: BN=yes, Aug=yes, Dropout=no)
    model_map = {
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet44': ResNet44,
        'resnet56': ResNet56,
        'resnet74': ResNet74,
        'resnet110': ResNet110
    }
    model = model_map[args.arch](num_classes=10, use_batchnorm=True, use_dropout=False)
    model = model.to(device)

    print(f"\nModel: {args.arch.upper()}")
    print(f"Configuration: BN=yes, Aug=standard (Crop+HFlip), Dropout=no, Optimizer=AdamW")
    print(f"Target: 99% train accuracy with standard augmentation")
    print(f"LR: {args.lr}, Weight Decay: {args.weight_decay}, Grad Clip: {args.grad_clip}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir
    )

    # Create evaluation loader (no augmentation, no shuffle)
    eval_loader = get_eval_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir
    )

    # Setup optimizer (AdamW with decoupled weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Setup learning rate scheduler: warmup (10) + cosine (490)
    # Continuous decay over full 500 epochs, no constant LR phase
    warmup_epochs = 10
    cosine_epochs = 490
    eta_min = 1e-6  # Very low minimum LR for full convergence to 99%

    # Warmup scheduler: linear increase from 1e-6 to args.lr over warmup_epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.001,  # Start at 0.1% of base LR (1e-6)
        end_factor=1.0,      # Reach full base LR (1e-3)
        total_iters=warmup_epochs
    )

    # Cosine annealing scheduler: decay from args.lr to eta_min over 490 epochs
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min
    )

    # Chain warmup + cosine schedulers
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Scheduler runs for full 500 epochs (no constant phase)
    scheduler_end_epoch = warmup_epochs + cosine_epochs  # = 500

    # Loss function (standard cross-entropy, no label smoothing for 99% train acc target)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    mi_history = []
    train_acc_aug_history = []   # Train acc on augmented data (for monitoring)
    train_acc_clean_history = []  # Train acc on clean data (for gen gap)
    test_acc_history = []
    gen_gap_history = []
    epochs_evaluated = []

    # Training loop
    print(f"\nStarting training for up to {args.epochs} epochs...")
    print(f"Target clean train accuracy for early stopping: {args.target_train_acc:.2f}%")
    print("="*70)

    final_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        # Train without CutMix to allow 99% train accuracy
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            cutmix_alpha=0.0, grad_clip=args.grad_clip
        )

        # Step scheduler for full 500 epochs (warmup + cosine decay to eta_min)
        # LR decays continuously from 1e-3 to 1e-6, no constant phase
        if epoch <= scheduler_end_epoch:
            scheduler.step()

        # Evaluate clean train accuracy every epoch for early stopping
        train_acc_clean = evaluate_accuracy(model, eval_loader, device)

        # Evaluate MI and accuracy only at specified intervals
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            # Evaluate test accuracy
            test_acc = evaluate_accuracy(model, test_loader, device)
            gen_gap = train_acc_clean - test_acc

            # Print progress
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Acc (aug): {train_acc:.2f}%, "
                  f"Train Acc (clean): {train_acc_clean:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%, "
                  f"Gen Gap: {gen_gap:.2f}%")

            # Evaluate MI (skip at final epoch since we do comprehensive MI eval after loop)
            if epoch != args.epochs:
                print(f"  Evaluating MI (n_masks={args.n_masks_train}, max_batches={args.max_eval_batches_train})...")
                mi_full, mean_mi_masked, mi_diff = evaluate_first_layer_mi(
                    model, eval_loader, device,
                    n_subsets=args.n_masks_train,
                    seed=args.seed + epoch,  # Different seed each time
                    max_batches=args.max_eval_batches_train
                )
                print(f"  MI: {mi_full:.6f}, MI_masked: {mean_mi_masked:.6f}, MI_diff: {mi_diff:.6f}")

                mi_history.append(mi_diff)
                train_acc_aug_history.append(train_acc)
                train_acc_clean_history.append(train_acc_clean)
                test_acc_history.append(test_acc)
                gen_gap_history.append(gen_gap)
                epochs_evaluated.append(epoch)

        # Check for early stopping
        if train_acc_clean >= args.target_train_acc:
            print(f"\n✓ Target clean train accuracy reached: {train_acc_clean:.4f}% >= {args.target_train_acc:.2f}%")
            print(f"Stopping training at epoch {epoch}")
            final_epoch = epoch
            break

    print("\n" + "="*70)
    print("Training completed!")

    # Final MI evaluation with more masks and batches
    print(f"\nFinal MI evaluation (n_masks={args.n_masks_final}, max_batches={args.max_eval_batches_final})...")
    final_mi_full, final_mean_mi_masked, final_mi_diff = evaluate_first_layer_mi(
        model, eval_loader, device,
        n_subsets=args.n_masks_final,
        seed=args.seed,
        max_batches=args.max_eval_batches_final
    )

    print(f"Final MI: {final_mi_full:.6f}")
    print(f"Final MI_masked: {final_mean_mi_masked:.6f}")
    print(f"Final MI_diff: {final_mi_diff:.6f}")

    # Save final checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{args.arch}_seed{args.seed}_final.pt"

    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch': args.arch,
        'seed': args.seed,
        'use_batchnorm': True,
        'use_augmentation': True,  # Standard (Crop+HFlip)
        'use_dropout': False,
        'optimizer_name': 'adamw',
        'train_acc_aug': train_acc_aug_history[-1] if train_acc_aug_history else train_acc,
        'train_acc_clean': train_acc_clean_history[-1] if train_acc_clean_history else train_acc_clean,
        'test_acc': test_acc_history[-1] if test_acc_history else test_acc,
        'gen_gap': gen_gap_history[-1] if gen_gap_history else gen_gap,
    }, checkpoint_path)

    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{args.arch}_seed{args.seed}_results.npz"

    np.savez(
        results_path,
        epochs_evaluated=np.array(epochs_evaluated),
        mi_history=np.array(mi_history),
        train_acc_aug_history=np.array(train_acc_aug_history),
        train_acc_clean_history=np.array(train_acc_clean_history),
        test_acc_history=np.array(test_acc_history),
        gen_gap_history=np.array(gen_gap_history),
        final_mi_full=final_mi_full,
        final_mean_mi_masked=final_mean_mi_masked,
        final_mi_diff=final_mi_diff,
        final_train_acc_aug=train_acc,
        final_train_acc_clean=train_acc_clean,
        final_test_acc=test_acc,
        final_gen_gap=gen_gap,
        arch=args.arch,
        seed=args.seed,
        use_batchnorm=True,
        use_augmentation=True,  # Standard (Crop+HFlip)
        use_dropout=False,
        optimizer='adamw',
    )

    print(f"Results saved to: {results_path}")

    # Also save detailed JSON
    json_path = results_dir / f"{args.arch}_seed{args.seed}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'arch': args.arch,
            'seed': args.seed,
            'optimizer': 'adamw',
            'use_batchnorm': True,
            'use_augmentation': True,  # Standard (Crop+HFlip)
            'use_dropout': False,
            'epochs': args.epochs,
            'final_epoch': final_epoch,
            'final_train_acc_aug': float(train_acc),
            'final_train_acc_clean': float(train_acc_clean),
            'final_test_acc': float(test_acc),
            'final_gen_gap': float(gen_gap),
            'final_mi_full': float(final_mi_full),
            'final_mean_mi_masked': float(final_mean_mi_masked),
            'final_mi_diff': float(final_mi_diff),
            'epochs_evaluated': [int(e) for e in epochs_evaluated],
            'mi_history': [float(m) for m in mi_history],
            'train_acc_aug_history': [float(a) for a in train_acc_aug_history],
            'train_acc_clean_history': [float(a) for a in train_acc_clean_history],
            'test_acc_history': [float(a) for a in test_acc_history],
            'gen_gap_history': [float(g) for g in gen_gap_history],
        }, f, indent=2)

    print(f"Detailed results saved to: {json_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()

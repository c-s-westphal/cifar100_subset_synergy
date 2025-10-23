"""
Visualize relationship between MI difference and generalization gap.

Creates a scatter plot with generalization gap on x-axis and MI difference on y-axis,
with different colors representing different architectures (VGG or ResNet).
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def get_arch_config(arch_family: str) -> Tuple[List[str], Dict[str, str], Dict[str, str], str]:
    """Get architecture configuration for a given family.

    Returns:
        (architectures, colors, labels, family_name)
    """
    if arch_family == 'vgg':
        architectures = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
        colors = {
            'vgg9': '#1f77b4',
            'vgg11': '#ff7f0e',
            'vgg13': '#2ca02c',
            'vgg16': '#d62728',
            'vgg19': '#9467bd'
        }
        labels = {
            'vgg9': 'VGG9',
            'vgg11': 'VGG11',
            'vgg13': 'VGG13',
            'vgg16': 'VGG16',
            'vgg19': 'VGG19'
        }
        family_name = 'VGG'
    elif arch_family == 'resnet':
        architectures = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']
        colors = {
            'resnet20': '#1f77b4',
            'resnet32': '#ff7f0e',
            'resnet44': '#2ca02c',
            'resnet56': '#d62728',
            'resnet110': '#9467bd'
        }
        labels = {
            'resnet20': 'ResNet-20',
            'resnet32': 'ResNet-32',
            'resnet44': 'ResNet-44',
            'resnet56': 'ResNet-56',
            'resnet110': 'ResNet-110'
        }
        family_name = 'ResNet'
    else:
        raise ValueError(f"Unknown architecture family: {arch_family}")

    return architectures, colors, labels, family_name


def load_final_results(results_dir: str, arch_family: str) -> Dict[str, List[dict]]:
    """Load final results organized by architecture.

    Args:
        results_dir: Path to results directory
        arch_family: Architecture family ('vgg' or 'resnet')

    Returns:
        Dict mapping arch -> list of result dicts (one per seed)
    """
    results_path = Path(results_dir)
    all_results = {}

    architectures, _, _, _ = get_arch_config(arch_family)

    # Configure seeds per architecture
    # VGG9 uses 10 seeds (0-9), others use 3 seeds (0-2)
    arch_seeds = {
        'vgg9': list(range(10)),  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    default_seeds = [0, 1, 2]

    for arch in architectures:
        all_results[arch] = []
        seeds = arch_seeds.get(arch, default_seeds)

        for seed in seeds:
            json_file = results_path / f"{arch}_seed{seed}_results.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_results[arch].append({
                        'seed': seed,
                        'gen_gap': data['final_gen_gap'],
                        'mi_diff': data['final_mi_diff'],
                        'test_acc': data['final_test_acc'],
                        'train_acc': data['final_train_acc_clean']
                    })
            else:
                print(f"Warning: {json_file} not found")

    return all_results


def plot_mi_vs_gengap(results: Dict[str, List[dict]], output_path: Path, arch_family: str):
    """Create scatter plot of MI difference vs generalization gap.

    Args:
        results: Results dictionary
        output_path: Path to save plot
        arch_family: Architecture family ('vgg' or 'resnet')
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    architectures, colors, labels, family_name = get_arch_config(arch_family)

    # Plot each architecture
    for arch in architectures:
        if arch not in results or not results[arch]:
            continue

        gen_gaps = [r['gen_gap'] for r in results[arch]]
        mi_diffs = [r['mi_diff'] for r in results[arch]]

        # Plot individual points
        ax.scatter(gen_gaps, mi_diffs,
                  color=colors[arch],
                  label=labels[arch],
                  s=150,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1.5)

        # Plot mean point with larger marker
        mean_gap = np.mean(gen_gaps)
        mean_mi = np.mean(mi_diffs)
        ax.scatter([mean_gap], [mean_mi],
                  color=colors[arch],
                  s=300,
                  marker='*',
                  edgecolors='black',
                  linewidth=2,
                  zorder=5)

    ax.set_xlabel('Generalization Gap (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MI Difference (bits)', fontsize=14, fontweight='bold')
    ax.set_title(f'MI Difference vs Generalization Gap\nAcross {family_name} Architectures',
                fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend1 = ax.legend(loc='best', fontsize=12, framealpha=0.9)

    # Add note about markers
    ax.text(0.02, 0.98, 'Circles: individual seeds\nStars: mean across seeds',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate MI vs Gen Gap visualization."""
    parser = argparse.ArgumentParser(description='Plot MI difference vs generalization gap')
    parser.add_argument('--arch', type=str, default='vgg', choices=['vgg', 'resnet'],
                       help='Architecture family to plot (vgg or resnet)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    args = parser.parse_args()

    print(f"Loading {args.arch.upper()} results...")
    results = load_final_results(args.results_dir, args.arch)

    # Create plots directory
    plots_dir = Path(args.output_dir)
    plots_dir.mkdir(exist_ok=True)

    output_file = plots_dir / f'mi_diff_vs_gen_gap_{args.arch}.png'
    print(f"Generating scatter plot for {args.arch.upper()}...")
    plot_mi_vs_gengap(results, output_file, args.arch)

    print("\n" + "="*70)
    print("Plot generated successfully!")
    print(f"Plot saved to: {output_file.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()

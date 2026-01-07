#!/usr/bin/env python3
"""
Compare memory usage across multiple configurations

Usage:
    python scripts/compare_configs.py config1.yaml config2.yaml config3.yaml
"""
import argparse
import sys
from pathlib import Path
import yaml
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.estimate import (
    load_config,
    create_args_from_config,
    create_transformer_config
)
from megatron_memory_estimator.estimate_013 import estimate_from_config


def extract_summary(config_dict, reports):
    """Extract summary information from config and reports"""
    model_cfg = config_dict['model']
    parallel_cfg = config_dict['parallelism']
    training_cfg = config_dict['training']

    # Get peak memory (from first PP stage, as they should be similar in most cases)
    peak_memory = reports[0]['total_gb']
    weight_grad_optim = reports[0]['weight_grad_optim_gb']
    activation = reports[0]['activation_gb']
    parameters = reports[0]['parameters_b']

    # Model description
    if model_cfg.get('num_experts'):
        model_desc = f"{model_cfg['num_layers']}L-{model_cfg['hidden_size']}H-{model_cfg['num_experts']}E"
    else:
        model_desc = f"{model_cfg['num_layers']}L-{model_cfg['hidden_size']}H-Dense"

    # Parallelism description
    parallel_desc = f"TP{parallel_cfg['tensor_model_parallel_size']}"
    parallel_desc += f"×PP{parallel_cfg['pipeline_model_parallel_size']}"
    if model_cfg.get('num_experts'):
        parallel_desc += f"×EP{parallel_cfg['expert_model_parallel_size']}"
    if parallel_cfg.get('context_parallel_size', 1) > 1:
        parallel_desc += f"×CP{parallel_cfg['context_parallel_size']}"

    return {
        'Model': model_desc,
        'Parallelism': parallel_desc,
        'GPUs': training_cfg['world_size'],
        'Batch': training_cfg['micro_batch_size'],
        'SeqLen': training_cfg['seq_length'],
        'Params (B)': f"{parameters:.2f}",
        'Weights+Opt (GB)': f"{weight_grad_optim:.2f}",
        'Activation (GB)': f"{activation:.2f}",
        'Total (GB)': f"{peak_memory:.2f}",
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare memory estimates across multiple configurations'
    )
    parser.add_argument('configs', nargs='+',
                        help='Paths to configuration files to compare')
    parser.add_argument('--output', type=str,
                        help='Save comparison table to file')
    parser.add_argument('--format', choices=['plain', 'grid', 'markdown', 'latex'],
                        default='grid',
                        help='Output table format')

    args = parser.parse_args()

    print("\nComparing configurations...")
    print("="*80)

    summaries = []

    for config_path in args.configs:
        print(f"\nProcessing: {config_path}")

        # Load and process configuration
        config_dict = load_config(config_path)
        megatron_args = create_args_from_config(config_dict)
        transformer_config = create_transformer_config(config_dict, megatron_args)

        # Run estimation
        reports, _ = estimate_from_config(transformer_config, megatron_args)

        # Extract summary
        summary = extract_summary(config_dict, reports)
        summary['Config File'] = Path(config_path).name
        summaries.append(summary)

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")

    # Prepare table
    headers = ['Config File', 'Model', 'Parallelism', 'GPUs', 'Batch', 'SeqLen',
               'Params (B)', 'Weights+Opt (GB)', 'Activation (GB)', 'Total (GB)']

    table_data = []
    for summary in summaries:
        row = [summary.get(h, '') for h in headers]
        table_data.append(row)

    # Print table
    table_format = args.format if args.format != 'plain' else 'simple'
    table_str = tabulate(table_data, headers=headers, tablefmt=table_format)
    print(table_str)
    print()

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Megatron Memory Estimation Comparison\n\n")
            f.write(table_str)
            f.write("\n")
        print(f"Comparison saved to: {args.output}\n")

    # Find most memory efficient configuration
    min_memory_idx = min(range(len(summaries)),
                         key=lambda i: float(summaries[i]['Total (GB)']))
    max_memory_idx = max(range(len(summaries)),
                         key=lambda i: float(summaries[i]['Total (GB)']))

    print("Summary:")
    print(f"  Most memory efficient: {summaries[min_memory_idx]['Config File']} "
          f"({summaries[min_memory_idx]['Total (GB)']} GB)")
    print(f"  Highest memory usage: {summaries[max_memory_idx]['Config File']} "
          f"({summaries[max_memory_idx]['Total (GB)']} GB)")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())

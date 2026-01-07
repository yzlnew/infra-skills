#!/usr/bin/env python3
"""
Validate YAML configuration files without running full estimation

Usage:
    python scripts/validate_config.py configs/examples/mixtral_8x7b.yaml
"""
import argparse
import yaml
from pathlib import Path


def validate_config(config_dict):
    """Validate configuration structure and values"""
    errors = []
    warnings = []

    # Check required sections
    required_sections = ['model', 'parallelism', 'training']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")
            return errors, warnings

    model = config_dict['model']
    parallel = config_dict['parallelism']
    training = config_dict['training']

    # Validate model config
    required_model_keys = ['num_layers', 'hidden_size', 'num_attention_heads',
                          'ffn_hidden_size', 'vocab_size', 'max_position_embeddings']
    for key in required_model_keys:
        if key not in model:
            errors.append(f"Missing required model key: {key}")

    # Check MoE consistency
    if model.get('num_experts'):
        if not model.get('moe_router_topk'):
            warnings.append("MoE enabled but moe_router_topk not specified, using default")
        if model['num_experts'] % parallel.get('expert_model_parallel_size', 1) != 0:
            errors.append(f"num_experts ({model['num_experts']}) must be divisible by "
                         f"expert_model_parallel_size ({parallel.get('expert_model_parallel_size', 1)})")

    # Validate parallelism
    required_parallel_keys = ['tensor_model_parallel_size', 'pipeline_model_parallel_size']
    for key in required_parallel_keys:
        if key not in parallel:
            errors.append(f"Missing required parallelism key: {key}")

    # Calculate and validate world size
    tp = parallel.get('tensor_model_parallel_size', 1)
    pp = parallel.get('pipeline_model_parallel_size', 1)
    ep = parallel.get('expert_model_parallel_size', 1)
    etp = parallel.get('expert_tensor_parallel_size', 1)
    cp = parallel.get('context_parallel_size', 1)

    required_gpus = tp * pp * ep * etp * cp
    specified_world_size = training.get('world_size', 1)

    if required_gpus > specified_world_size:
        errors.append(f"Parallelism requires {required_gpus} GPUs but world_size is {specified_world_size}")
    elif required_gpus < specified_world_size:
        dp = specified_world_size // required_gpus
        warnings.append(f"Data parallelism size will be {dp} (world_size / (tp*pp*ep*etp*cp))")

    # Validate training config
    required_training_keys = ['micro_batch_size', 'seq_length', 'world_size']
    for key in required_training_keys:
        if key not in training:
            errors.append(f"Missing required training key: {key}")

    # Check recompute consistency
    if training.get('recompute_granularity'):
        if training['recompute_granularity'] not in [None, 'full', 'selective']:
            errors.append(f"Invalid recompute_granularity: {training['recompute_granularity']}")
        if training.get('recompute_method') and training['recompute_method'] not in ['uniform', 'block']:
            errors.append(f"Invalid recompute_method: {training['recompute_method']}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description='Validate Megatron configuration files'
    )
    parser.add_argument('configs', nargs='+',
                       help='Configuration files to validate')

    args = parser.parse_args()

    all_valid = True

    for config_path in args.configs:
        print(f"\nValidating: {config_path}")
        print("-" * 60)

        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            errors, warnings = validate_config(config_dict)

            if errors:
                all_valid = False
                print("❌ ERRORS:")
                for error in errors:
                    print(f"  - {error}")

            if warnings:
                print("⚠️  WARNINGS:")
                for warning in warnings:
                    print(f"  - {warning}")

            if not errors and not warnings:
                print("✅ Configuration is valid!")

            # Print summary
            print(f"\nConfiguration Summary:")
            model = config_dict['model']
            parallel = config_dict['parallelism']
            training = config_dict['training']

            if model.get('num_experts'):
                print(f"  Model: {model['num_layers']}L-{model['hidden_size']}H-{model['num_experts']}E "
                     f"(top-{model.get('moe_router_topk', 2)})")
            else:
                print(f"  Model: {model['num_layers']}L-{model['hidden_size']}H (Dense)")

            print(f"  Parallelism: TP={parallel.get('tensor_model_parallel_size', 1)}, "
                 f"PP={parallel.get('pipeline_model_parallel_size', 1)}, "
                 f"EP={parallel.get('expert_model_parallel_size', 1)}, "
                 f"CP={parallel.get('context_parallel_size', 1)}")
            print(f"  Training: BS={training.get('micro_batch_size')}, "
                 f"SeqLen={training.get('seq_length')}, "
                 f"GPUs={training.get('world_size')}")

        except Exception as e:
            all_valid = False
            print(f"❌ ERROR: Failed to load or parse configuration: {e}")

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All configurations are valid!")
        return 0
    else:
        print("❌ Some configurations have errors")
        return 1


if __name__ == '__main__':
    exit(main())

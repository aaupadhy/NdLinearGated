import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import yaml
import json
from pathlib import Path

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_accuracy_curves(results_csv, output_dir):
    df = pd.read_csv(results_csv)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        plt.figure(figsize=(12, 8))
        
        for model_type in ['NdLinear', 'NdLinearGated']:
            model_df = dataset_df[dataset_df['model'] == model_type]
            
            if model_type == 'NdLinear':
                plt.scatter(1, model_df['accuracy'].values[0], s=100, marker='*', color='red', 
                           label=f'{model_type}')
            else:
                for _, row in model_df.iterrows():
                    gating_mode = row['gating_mode']
                    gating_hidden_dim = row['gating_hidden_dim']
                    gated_modes = row['gated_modes']
                    accuracy = row['accuracy']
                    
                    plt.scatter(gating_hidden_dim, accuracy, s=100, 
                               label=f'{model_type}-{gating_mode}-{gated_modes}')
        
        plt.title(f'Accuracy vs Gating Dimension on {dataset}')
        plt.xlabel('Gating Hidden Dimension')
        plt.ylabel('Accuracy (%)')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        out_path = f"{output_dir}/gating_capacity/accuracy_comparison_{dataset}.png"
        ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def plot_entropy_vs_epoch(manifest_yaml, output_dir):
    with open(manifest_yaml, 'r') as f:
        manifest = yaml.safe_load(f)
    
    datasets = set()
    model_configs = {}
    
    for exp_name, exp_data in manifest.items():
        if exp_data['model'] == 'NdLinearGated':
            dataset = exp_data['dataset']
            datasets.add(dataset)
            
            config_key = f"{exp_data.get('gating_mode', 'soft')}_{exp_data.get('gated_modes', 'all')}_{exp_data.get('gating_hidden_dim', 16)}"
            
            if dataset not in model_configs:
                model_configs[dataset] = {}
            
            model_configs[dataset][config_key] = exp_data['plots']['entropy_curve']
    
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        
        for config_key, plot_path in model_configs.get(dataset, {}).items():
            if plot_path and os.path.exists(plot_path):
                try:
                    img = plt.imread(plot_path)
                    plt.figure(figsize=(14, 10))
                    plt.imshow(img)
                    plt.axis('off')
                    
                    out_path = f"{output_dir}/gating_capacity/entropy_vs_epoch_{dataset}_{config_key}.png"
                    ensure_dir(out_path)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Error processing {plot_path}: {e}")
    
    for dataset in datasets:
        plt.figure(figsize=(14, 10))
        for idx, (config_key, plot_path) in enumerate(model_configs.get(dataset, {}).items()):
            if plot_path and os.path.exists(plot_path):
                try:
                    plt.subplot(2, 3, idx+1)
                    img = plt.imread(plot_path)
                    plt.imshow(img)
                    plt.title(config_key)
                    plt.axis('off')
                except Exception as e:
                    print(f"Error processing {plot_path}: {e}")
        
        plt.tight_layout()
        out_path = f"{output_dir}/gating_capacity/entropy_comparison_{dataset}.png"
        ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def plot_heatmaps_over_time(manifest_yaml, output_dir):
    with open(manifest_yaml, 'r') as f:
        manifest = yaml.safe_load(f)
    
    for exp_name, exp_data in manifest.items():
        if exp_data['model'] == 'NdLinearGated':
            dataset = exp_data['dataset']
            gating_mode = exp_data.get('gating_mode', 'soft')
            gated_modes = exp_data.get('gated_modes', 'all')
            gating_hidden_dim = exp_data.get('gating_hidden_dim', 16)
            
            config_key = f"{gating_mode}_{gated_modes}_{gating_hidden_dim}"
            
            heatmap_dir = f"{output_dir}/plots"
            if os.path.exists(heatmap_dir):
                heatmap_files = [f for f in os.listdir(heatmap_dir) if 
                               f.startswith(f"NdLinearGated_{config_key}_{dataset}_gate_heatmap") and 
                               f.endswith(".png")]
                
                modes = {}
                for file in heatmap_files:
                    parts = file.split('_')
                    mode_idx = parts[-3]
                    epoch = parts[-1].replace('.png', '')
                    
                    if mode_idx not in modes:
                        modes[mode_idx] = []
                    
                    modes[mode_idx].append((int(epoch), f"{heatmap_dir}/{file}"))
                
                for mode_idx, files in modes.items():
                    files.sort(key=lambda x: x[0])
                    
                    plt.figure(figsize=(16, 10))
                    for i, (epoch, file_path) in enumerate(files):
                        if i < 9:
                            plt.subplot(3, 3, i+1)
                            img = plt.imread(file_path)
                            plt.imshow(img)
                            plt.title(f"Epoch {epoch}")
                            plt.axis('off')
                    
                    plt.suptitle(f"{exp_data['model']} {config_key} on {dataset} - Mode {mode_idx} Evolution")
                    plt.tight_layout()
                    
                    out_path = f"{output_dir}/mode_gating/evolution_{dataset}_{config_key}_mode_{mode_idx}.png"
                    ensure_dir(out_path)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close()

def plot_gate_transfer_comparison(gate_transfer_dir, output_dir):
    if not os.path.exists(gate_transfer_dir):
        return
    
    transfer_dirs = [d for d in os.listdir(gate_transfer_dir) 
                    if os.path.isdir(f"{gate_transfer_dir}/{d}")]
    
    for transfer_dir in transfer_dirs:
        results_path = f"{gate_transfer_dir}/{transfer_dir}/results.yaml"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = yaml.safe_load(f)
            
            heatmap_files = [f for f in os.listdir(f"{gate_transfer_dir}/{transfer_dir}") 
                           if f.startswith("heatmap_mode_") and f.endswith(".png")]
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 1, 1)
            plt.bar(['Accuracy', 'Active Gates', 'Gate Entropy'], 
                   [results.get('accuracy', 0), 
                    results.get('active_gates', 0) * 100, 
                    results.get('gate_entropy', 0)])
            plt.title(f"Transfer Metrics: {results.get('source_dataset', '')} → {results.get('target_dataset', '')}")
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            
            for i, heatmap_file in enumerate(sorted(heatmap_files)):
                if i < 3:
                    plt.subplot(2, 3, i+4)
                    img_path = f"{gate_transfer_dir}/{transfer_dir}/{heatmap_file}"
                    img = plt.imread(img_path)
                    plt.imshow(img)
                    plt.title(f"Mode {i} Gate")
                    plt.axis('off')
            
            plt.tight_layout()
            out_path = f"{output_dir}/gate_transfer/summary_{transfer_dir}.png"
            ensure_dir(out_path)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()

def plot_proxy_compute_comparison(results_csv, output_dir):
    df = pd.read_csv(results_csv)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(dataset_df['proxy_compute'], dataset_df['accuracy'], 
                            s=100, c=dataset_df['percent_active_slices'], 
                            cmap='viridis', alpha=0.7)
        
        for i, row in dataset_df.iterrows():
            if row['model'] == 'NdLinear':
                label = 'NdLinear'
            else:
                label = f"{row['model']}-{row['gating_mode']}-{row['gated_modes']}"
            
            plt.annotate(label, (row['proxy_compute'], row['accuracy']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, label='% Active Slices')
        plt.title(f'Accuracy vs Computational Cost on {dataset}')
        plt.xlabel('Proxy Compute (operations)')
        plt.ylabel('Accuracy (%)')
        
        if max(dataset_df['proxy_compute']) / min(dataset_df['proxy_compute']) > 10:
            plt.xscale('log')
        
        plt.grid(True, alpha=0.3)
        
        out_path = f"{output_dir}/proxy_analysis/accuracy_vs_compute_{dataset}.png"
        ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def plot_profiler_comparison(profiler_dir, output_dir):
    if not os.path.exists(profiler_dir):
        return
    
    profiler_files = [f for f in os.listdir(profiler_dir) if f.endswith(".png")]
    
    dataset_files = {}
    for file in profiler_files:
        parts = file.split('_')
        dataset = parts[-1].replace('.png', '')
        
        if dataset not in dataset_files:
            dataset_files[dataset] = []
        
        dataset_files[dataset].append(file)
    
    for dataset, files in dataset_files.items():
        plt.figure(figsize=(20, 15))
        
        for i, file in enumerate(files):
            plt.subplot(len(files), 1, i+1)
            img_path = f"{profiler_dir}/{file}"
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title(file.replace('.png', ''))
            plt.axis('off')
        
        plt.tight_layout()
        out_path = f"{output_dir}/profiler_breakdown/comparison_{dataset}.png"
        ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def generate_all_visualizations(output_dir):
    results_csv = f"{output_dir}/results_table.csv"
    manifest_yaml = f"{output_dir}/experiment_manifest.yaml"
    gate_transfer_dir = f"{output_dir}/gate_transfer"
    profiler_dir = f"{output_dir}/profiler_breakdown"
    
    if os.path.exists(results_csv):
        plot_accuracy_curves(results_csv, output_dir)
        plot_proxy_compute_comparison(results_csv, output_dir)
    
    if os.path.exists(manifest_yaml):
        plot_entropy_vs_epoch(manifest_yaml, output_dir)
        plot_heatmaps_over_time(manifest_yaml, output_dir)
    
    if os.path.exists(gate_transfer_dir):
        plot_gate_transfer_comparison(gate_transfer_dir, output_dir)
    
    if os.path.exists(profiler_dir):
        plot_profiler_comparison(profiler_dir, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for NdLinearGated experiments")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory containing experiment outputs")
    
    args = parser.parse_args()
    generate_all_visualizations(args.output_dir) 
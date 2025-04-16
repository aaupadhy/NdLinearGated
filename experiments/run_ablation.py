import os
import sys
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import json
from datetime import datetime
import torch.profiler
from pathlib import Path
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ndlinear import NdLinear, NdLinearGated

def parse_args():
    parser = argparse.ArgumentParser(description="NdLinear vs NdLinearGated Ablation Study")
    parser.add_argument("--config", type=str, default="experiments/configs/ablation.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save outputs")
    parser.add_argument("--run_profiler", action="store_true", 
                        help="Whether to run the profiler")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class TextDatasetWrapper:
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'sentence' in item:
            text = item['sentence']
            label = item['label']
        elif 'text' in item:
            text = item['text']
            label = item['label']
        else:
            raise ValueError("Unsupported dataset format")
        
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', 
                                 truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

def get_dataset(dataset_name, batch_size):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = load_dataset("cifar10", split="train")
        test_dataset = load_dataset("cifar10", split="test")
        
        def transform_cifar(example):
            img = example['img'].convert('RGB')
            img_tensor = transform(img)
            return {"pixel_values": img_tensor, "label": example["label"]}
        
        train_dataset = train_dataset.map(transform_cifar, remove_columns=['img'])
        test_dataset = test_dataset.map(transform_cifar, remove_columns=['img'])
        
        train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
        test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        input_shape = (32, 32, 3)
        num_classes = 10
        
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = load_dataset("mnist", split="train")
        test_dataset = load_dataset("mnist", split="test")
        
        def transform_mnist(example):
            img = example['image'].convert('L')
            img_tensor = transform(img)
            return {"pixel_values": img_tensor, "label": example["label"]}
        
        train_dataset = train_dataset.map(transform_mnist, remove_columns=['image'])
        test_dataset = test_dataset.map(transform_mnist, remove_columns=['image'])
        
        train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
        test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        input_shape = (28, 28, 1)
        num_classes = 10
        
    elif dataset_name == "ag_news":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = load_dataset("ag_news", split="train")
        test_dataset = load_dataset("ag_news", split="test")
        
        train_dataset_wrapped = TextDatasetWrapper(train_dataset, tokenizer)
        test_dataset_wrapped = TextDatasetWrapper(test_dataset, tokenizer)
        
        train_loader = DataLoader(train_dataset_wrapped, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset_wrapped, batch_size=batch_size)
        
        input_shape = (128, 768)
        num_classes = 4
        
    elif dataset_name == "imdb":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = load_dataset("imdb", split="train")
        test_dataset = load_dataset("imdb", split="test")
        
        train_dataset_wrapped = TextDatasetWrapper(train_dataset, tokenizer, max_length=256)
        test_dataset_wrapped = TextDatasetWrapper(test_dataset, tokenizer, max_length=256)
        
        train_loader = DataLoader(train_dataset_wrapped, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset_wrapped, batch_size=batch_size)
        
        input_shape = (256, 768)
        num_classes = 2
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_loader, test_loader, input_shape, num_classes

class VisionModel(nn.Module):
    def __init__(self, model_type, input_shape, num_classes, **kwargs):
        super(VisionModel, self).__init__()
        self.model_type = model_type
        
        if model_type == "NdLinear":
            self.ndlayer = NdLinear(input_dims=input_shape, hidden_size=(64, 64, 16))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(64 * 64 * 16, num_classes)
        else:
            self.ndlayer = NdLinearGated(
                input_dims=input_shape, 
                hidden_size=(64, 64, 16),
                **kwargs
            )
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(64 * 64 * 16, num_classes)
    
    def forward(self, x):
        if len(x.shape) == 4: 
            x = x.permute(0, 2, 3, 1)
        x = self.ndlayer(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TextModel(nn.Module):
    def __init__(self, model_type, input_shape, num_classes, **kwargs):
        super(TextModel, self).__init__()
        self.model_type = model_type
        
        self.embedding = nn.Embedding(30522, 768)
        
        if model_type == "NdLinear":
            self.ndlayer = NdLinear(input_dims=input_shape, hidden_size=(32, 128))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, num_classes)
        else:
            self.ndlayer = NdLinearGated(
                input_dims=input_shape, 
                hidden_size=(32, 128),
                **kwargs
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.ndlayer(x)
        x = x.transpose(1, 2)  
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        return x

def build_model(model_type, dataset_name, input_shape, num_classes, **kwargs):
    if dataset_name in ["cifar10", "mnist"]:
        return VisionModel(model_type, input_shape, num_classes, **kwargs)
    else:
        return TextModel(model_type, input_shape, num_classes, **kwargs)

def compute_gate_entropy(model):
    if not hasattr(model, 'ndlayer') or not hasattr(model.ndlayer, 'gate_networks'):
        return 0.0
    
    entropies = []
    for gate_net in model.ndlayer.gate_networks:
        with torch.no_grad():
            device = next(gate_net.parameters()).device
            dummy_input = torch.randn(1, gate_net[0].in_features, device=device)
            gate_value = gate_net(dummy_input).item()
            
            gate_value = max(min(gate_value, 0.99), 0.01)
            
            entropy = -gate_value * np.log2(gate_value) - (1 - gate_value) * np.log2(1 - gate_value)
            entropies.append(entropy)
    
    return np.mean(entropies) if entropies else 0.0

def compute_active_gates(model):
    if not hasattr(model, 'ndlayer') or not hasattr(model.ndlayer, 'gate_networks'):
        return 0.0
    
    active_gates = 0
    total_gates = len(model.ndlayer.gate_networks)
    
    for gate_net in model.ndlayer.gate_networks:
        with torch.no_grad():
            device = next(gate_net.parameters()).device
            dummy_input = torch.randn(1, gate_net[0].in_features, device=device)
            gate_value = gate_net(dummy_input).item()
            
            if model.ndlayer.gating_mode == "hard":
                active_gates += 1 if gate_value > 0.5 else 0
            else:
                active_gates += gate_value
    
    return active_gates / total_gates if total_gates > 0 else 0.0

def compute_proxy_compute(model, input_shape):
    if not hasattr(model, 'ndlayer') or not hasattr(model.ndlayer, 'gate_networks'):
        total_compute = 0
        for i, layer in enumerate(model.ndlayer.align_layers):
            total_compute += layer.in_features * layer.out_features
        return total_compute
    
    total_compute = 0
    for i, (layer, gate_net) in enumerate(zip(model.ndlayer.align_layers, model.ndlayer.gate_networks)):
        with torch.no_grad():
            device = next(gate_net.parameters()).device
            dummy_input = torch.randn(1, gate_net[0].in_features, device=device)
            gate_value = gate_net(dummy_input).item()
            
            if model.ndlayer.gating_mode == "hard":
                factor = 1.0 if gate_value > 0.5 else 0.0
            else:
                factor = gate_value
            
            layer_compute = layer.in_features * layer.out_features
            total_compute += factor * layer_compute
    
    return total_compute

def train_epoch(model, train_loader, optimizer, criterion, device, dataset_name):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if dataset_name in ["cifar10", "mnist"]:
            data, target = batch["pixel_values"].to(device), batch["label"].to(device)
            output = model(data)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            output = model(input_ids, attention_mask)
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device, dataset_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if dataset_name in ["cifar10", "mnist"]:
                data, target = batch["pixel_values"].to(device), batch["label"].to(device)
                output = model(data)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                output = model(input_ids, attention_mask)
            
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def get_gate_heatmap_data(model):
    if not hasattr(model, 'ndlayer') or not hasattr(model.ndlayer, 'gate_networks'):
        return None
    
    heatmap_data = []
    for i, gate_net in enumerate(model.ndlayer.gate_networks):
        with torch.no_grad():
            device = next(gate_net.parameters()).device
            test_inputs = torch.linspace(-3, 3, 100, device=device).unsqueeze(1).repeat(1, gate_net[0].in_features)
            gate_values = []
            
            batch_size = 10
            for j in range(0, test_inputs.shape[0], batch_size):
                batch = test_inputs[j:j+batch_size]
                gate_value = gate_net(batch)
                gate_values.append(gate_value)
            
            gate_values = torch.cat(gate_values, dim=0).squeeze().cpu().numpy()
            heatmap_data.append(gate_values)
    
    return heatmap_data

def plot_accuracy_curve(epochs, train_accs, test_accs, exp_name, model_name, dataset_name, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accs, 'b-', label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accs, 'r-', label='Test Accuracy')
    plt.title(f"{model_name} on {dataset_name} - Accuracy vs Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    general_out_path = f"{output_dir}/plots/{model_name}_{dataset_name}_accuracy_curve.png"
    dataset_out_path = f"{output_dir}/{dataset_name}/plots/{model_name}_accuracy_curve.png"
    
    os.makedirs(os.path.dirname(general_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
    
    plt.savefig(general_out_path)
    plt.savefig(dataset_out_path)
    plt.close()

def plot_entropy_curve(epochs, entropies, exp_name, model_name, dataset_name, output_dir):
    if not entropies:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), entropies, 'g-')
    plt.title(f"{model_name} on {dataset_name} - Gate Entropy vs Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Gate Entropy (bits)')
    plt.grid(True)
    
    general_out_path = f"{output_dir}/plots/{model_name}_{dataset_name}_entropy_curve.png"
    dataset_out_path = f"{output_dir}/{dataset_name}/plots/{model_name}_entropy_curve.png"
    
    os.makedirs(os.path.dirname(general_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
    
    plt.savefig(general_out_path)
    plt.savefig(dataset_out_path)
    plt.close()

def plot_gate_heatmap(heatmap_data, epoch, exp_name, model_name, dataset_name, output_dir):
    if heatmap_data is None: 
        return
    
    for i, gate_values in enumerate(heatmap_data):
        plt.figure(figsize=(10, 6))
        plt.imshow(gate_values.reshape(10, 10), cmap='viridis', aspect='auto')
        plt.colorbar(label='Gate Value')
        plt.title(f"{model_name} on {dataset_name} - Gate Heatmap Mode {i}")
        plt.xlabel('Input Value')
        plt.ylabel('Input Value')
        
        general_out_path = f"{output_dir}/plots/{model_name}_{dataset_name}_gate_heatmap_mode_{i}_epoch_{epoch}.png"
        dataset_out_path = f"{output_dir}/{dataset_name}/plots/{model_name}_gate_heatmap_mode_{i}_epoch_{epoch}.png"
        
        os.makedirs(os.path.dirname(general_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
        
        plt.savefig(general_out_path)
        plt.savefig(dataset_out_path)
        plt.close()

def plot_active_gates(active_gates_per_mode, exp_name, model_name, dataset_name, output_dir):
    if not active_gates_per_mode:
        return
    
    plt.figure(figsize=(10, 6))
    modes = range(len(active_gates_per_mode))
    plt.bar(modes, active_gates_per_mode)
    plt.title(f"{model_name} on {dataset_name} - Active Gate Distribution")
    plt.xlabel('Mode')
    plt.ylabel('Gate Activation')
    plt.xticks(modes)
    plt.grid(True, axis='y')
    
    general_out_path = f"{output_dir}/plots/{model_name}_{dataset_name}_active_gate_distribution.png"
    dataset_out_path = f"{output_dir}/{dataset_name}/plots/{model_name}_active_gate_distribution.png"
    
    os.makedirs(os.path.dirname(general_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
    
    plt.savefig(general_out_path)
    plt.savefig(dataset_out_path)
    plt.close()

def plot_proxy_compute_vs_accuracy(proxy_computes, accuracies, model_names, exp_name, dataset_name, output_dir):
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating proxy compute vs accuracy plot for {dataset_name}")
    logger.info(f"Model names: {model_names}")
    logger.info(f"Proxy computes: {proxy_computes}")
    logger.info(f"Accuracies: {accuracies}")
    
    if not model_names or not proxy_computes or not accuracies:
        logger.warning("Cannot generate plot: missing data")
        return
    
    plt.figure(figsize=(10, 6))
    for model_name, proxy, acc in zip(model_names, proxy_computes, accuracies):
        plt.scatter(proxy, acc, label=model_name, s=100)
    
    plt.title(f"Models on {dataset_name} - Accuracy vs Compute")
    plt.xlabel('Proxy Compute (operations)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    non_zero_proxy_computes = [x for x in proxy_computes if x > 0]
    if non_zero_proxy_computes and max(non_zero_proxy_computes) / min(non_zero_proxy_computes) > 10:
        plt.xscale('log')
    
    general_out_path = f"{output_dir}/plots/models_{dataset_name}_proxy_compute_vs_accuracy.png"
    dataset_out_path = f"{output_dir}/{dataset_name}/plots/models_proxy_compute_vs_accuracy.png"
    
    os.makedirs(os.path.dirname(general_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
    
    plt.savefig(general_out_path)
    plt.savefig(dataset_out_path)
    plt.close()
    
    logger.info(f"Plot saved to {general_out_path} and {dataset_out_path}")

def save_summary(model_name, dataset_name, params, metrics, output_dir):
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        **params,
        **metrics
    }
    
    out_path = f"{output_dir}/summary_{model_name}_{dataset_name}.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    return out_path

def append_to_results_table(results_row, output_dir):
    csv_path = f"{output_dir}/results_table.csv"
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results_row)
    
    return csv_path

def update_experiment_manifest(experiment_data, output_dir):
    manifest_path = f"{output_dir}/experiment_manifest.yaml"
    
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}
    
    experiment_name = experiment_data["experiment_name"]
    manifest[experiment_name] = experiment_data
    
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    return manifest_path

def run_profiler(model, loader, device, dataset_name, output_dir, model_name):
    logger = logging.getLogger(__name__)
    
    def trace_handler(prof):
        prof.export_chrome_trace(f"{output_dir}/profile_timeline_{model_name}.json")
        
        top_ops = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        logger.info(f"Top 10 operations for {model_name}:")
        logger.info(top_ops)
        
        events = prof.key_averages()
        op_names = []
        op_times = []
        for evt in events:
            if len(op_names) < 10:
                op_names.append(evt.key)
                op_times.append(evt.cpu_time_total)
        
        op_times, op_names = zip(*sorted(zip(op_times, op_names), reverse=True))
        
        plt.figure(figsize=(15, 8))
        y_pos = np.arange(len(op_names))
        plt.barh(y_pos, op_times, align='center')
        plt.yticks(y_pos, op_names)
        plt.xlabel('Total CPU Time (μs)')
        plt.title(f'Top Operations by Cumulative Time - {model_name}')
        
        plt.savefig(f"{output_dir}/profiler_breakdown/{model_name}_{dataset_name}.png", bbox_inches='tight')
        plt.close()
    
    model.eval()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= 9:
                    break
                
                if dataset_name in ["cifar10", "mnist"]:
                    data = batch["pixel_values"].to(device)
                    output = model(data)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    output = model(input_ids, attention_mask)
                
                prof.step()

def perform_gate_transfer(source_model, target_train_loader, target_test_loader, device, 
                          source_dataset, target_dataset, input_shape, num_classes, output_dir):
    logger = logging.getLogger(__name__)
    
    if not hasattr(source_model, 'ndlayer') or not hasattr(source_model.ndlayer, 'gate_networks'):
        logger.info("Skipping gate transfer - source model does not have gate networks")
        return None
    
    model_type = source_model.model_type
    
    gating_mode = source_model.ndlayer.gating_mode
    gating_hidden_dim = source_model.ndlayer.gating_hidden_dim
    gated_modes = source_model.ndlayer.gated_modes
    
    target_model = build_model(
        model_type, 
        target_dataset, 
        input_shape, 
        num_classes,
        gating_mode=gating_mode, 
        gating_hidden_dim=gating_hidden_dim, 
        gated_modes=gated_modes
    )
    
    for i, gate_net in enumerate(source_model.ndlayer.gate_networks):
        if i < len(target_model.ndlayer.gate_networks):
            for param, source_param in zip(target_model.ndlayer.gate_networks[i].parameters(), 
                                         gate_net.parameters()):
                param.data.copy_(source_param.data)
                param.requires_grad = False
    
    optimizer = optim.Adam([p for n, p in target_model.named_parameters() 
                           if 'gate_networks' not in n], lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    target_model.to(device)
    
    epochs = 5
    best_acc = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(target_model, target_train_loader, optimizer, criterion, 
                                          device, target_dataset)
        test_loss, test_acc = test(target_model, target_test_loader, criterion, device, target_dataset)
        
        logger.info(f"Gate Transfer {source_dataset} -> {target_dataset} | "
              f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            
            heatmap_data = get_gate_heatmap_data(target_model)
            
            for i, gate_values in enumerate(heatmap_data):
                plt.figure(figsize=(10, 6))
                plt.imshow(gate_values.reshape(10, 10), cmap='viridis', aspect='auto')
                plt.colorbar(label='Gate Value')
                plt.title(f"Gate Transfer {source_dataset} -> {target_dataset} - Mode {i}")
                
                out_path = f"{output_dir}/gate_transfer/{source_dataset}_to_{target_dataset}/heatmap_mode_{i}.png"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plt.savefig(out_path)
                plt.close()
    
    active_gates = compute_active_gates(target_model)
    gate_entropy = compute_gate_entropy(target_model)
    
    transfer_results = {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'accuracy': best_acc,
        'active_gates': active_gates,
        'gate_entropy': gate_entropy
    }
    
    out_path = f"{output_dir}/gate_transfer/{source_dataset}_to_{target_dataset}/results.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        yaml.dump(transfer_results, f, default_flow_style=False)
    
    logger.info(f"Gate transfer completed. Best accuracy: {best_acc:.2f}%")
    return transfer_results

def is_experiment_completed(exp_name, dataset_name, output_dir):
    summary_path = f"{output_dir}/summary_{exp_name}_{dataset_name}.yaml"
    return os.path.exists(summary_path)

def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ablation study")
    logger.info(f"Using config file: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        os.makedirs(f"{output_dir}/{dataset_name}", exist_ok=True)
        os.makedirs(f"{output_dir}/{dataset_name}/plots", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    dataset_dict = {}
    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        batch_size = dataset_config["batch_size"]
        
        logger.info(f"Loading {dataset_name} dataset...")
        train_loader, test_loader, input_shape, num_classes = get_dataset(dataset_name, batch_size)
        
        dataset_dict[dataset_name] = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
    
    for dataset_name, dataset_info in dataset_dict.items():
        logger.info(f"\n=== Running experiments on {dataset_name} dataset ===")
        
        model_results = []
        
        for model_config in config["model_configs"]:
            model_name = model_config["name"]
            model_params = model_config["params"]
            
            if model_name == "NdLinear":
                exp_name = f"{model_name}_{dataset_name}"
                param_suffix = ""
            else:
                gating_mode = model_params.get("gating_mode", "soft")
                gating_hidden_dim = model_params.get("gating_hidden_dim", 16)
                gated_modes = model_params.get("gated_modes", "all")
                
                exp_name = f"{model_name}_{gating_mode}_{gated_modes}_{gating_hidden_dim}_{dataset_name}"
                param_suffix = f"_{gating_mode}_{gated_modes}_{gating_hidden_dim}"
            
            if is_experiment_completed(exp_name, dataset_name, output_dir):
                logger.info(f"Skipping {exp_name} - already completed")
                continue
            
            logger.info(f"\nRunning {exp_name}")
            
            model = build_model(
                model_name, 
                dataset_name, 
                dataset_info["input_shape"], 
                dataset_info["num_classes"], 
                **model_params
            )
            model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], 
                                  weight_decay=config["training"]["weight_decay"])
            criterion = nn.CrossEntropyLoss()
            
            train_accs = []
            test_accs = []
            entropies = []
            active_gates_history = []
            
            best_acc = 0
            patience = config["training"]["early_stopping_patience"]
            patience_counter = 0
            
            epochs = config["training"]["epochs"]
            start_time = time.time()
            peak_memory = 0
            
            for epoch in range(epochs):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                train_loss, train_acc = train_epoch(
                    model, 
                    dataset_info["train_loader"], 
                    optimizer, 
                    criterion, 
                    device, 
                    dataset_name
                )
                
                test_loss, test_acc = test(
                    model, 
                    dataset_info["test_loader"], 
                    criterion, 
                    device, 
                    dataset_name
                )
                
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                
                if model_name == "NdLinearGated":
                    entropy = compute_gate_entropy(model)
                    active_gates = compute_active_gates(model)
                    entropies.append(entropy)
                    active_gates_history.append(active_gates)
                    
                    gate_status = f"Gate Entropy: {entropy:.4f}, Active Gates: {active_gates:.2f}"
                else:
                    gate_status = ""
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, "
                          f"Test Acc: {test_acc:.2f}% | {gate_status}")
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    patience_counter = 0
                    
                    if model_name == "NdLinearGated" and epoch % config["training"]["save_every_n_epochs"] == 0:
                        heatmap_data = get_gate_heatmap_data(model)
                        plot_gate_heatmap(heatmap_data, epoch+1, exp_name, model_name + param_suffix, 
                                         dataset_name, output_dir)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                if torch.cuda.is_available():
                    current_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_memory = max(peak_memory, current_peak)
            
            train_time = time.time() - start_time
            
            active_gates_per_mode = []
            if model_name == "NdLinearGated":
                for i, gate_net in enumerate(model.ndlayer.gate_networks):
                    with torch.no_grad():
                        device = next(gate_net.parameters()).device
                        dummy_input = torch.randn(1, gate_net[0].in_features, device=device)
                        gate_value = gate_net(dummy_input).item()
                        active_gates_per_mode.append(gate_value)
            
            proxy_compute = compute_proxy_compute(model, dataset_info["input_shape"])
            
            plot_accuracy_curve(min(epoch+1, epochs), train_accs, test_accs, exp_name, 
                               model_name + param_suffix, dataset_name, output_dir)
            
            if model_name == "NdLinearGated":
                plot_entropy_curve(min(epoch+1, epochs), entropies, exp_name, 
                                  model_name + param_suffix, dataset_name, output_dir)
                plot_active_gates(active_gates_per_mode, exp_name, model_name + param_suffix, 
                                 dataset_name, output_dir)
            
            if args.run_profiler and dataset_name == "cifar10":
                logger.info(f"Running profiler for {model_name} on {dataset_name}...")
                run_profiler(model, dataset_info["test_loader"], device, dataset_name, 
                            output_dir, model_name + param_suffix)
            
            if model_name == "NdLinearGated":
                final_entropy = entropies[-1] if entropies else 0
                final_active_gates = active_gates_history[-1] if active_gates_history else 0
            else:
                final_entropy = 0
                final_active_gates = 1.0
            
            results = {
                "model": model_name,
                "dataset": dataset_name,
                "gating_mode": model_params.get("gating_mode", "N/A") if model_name == "NdLinearGated" else "N/A",
                "gating_hidden_dim": model_params.get("gating_hidden_dim", 0) if model_name == "NdLinearGated" else 0,
                "gated_modes": model_params.get("gated_modes", "N/A") if model_name == "NdLinearGated" else "N/A",
                "accuracy": best_acc,
                "train_time": train_time,
                "peak_memory_mb": peak_memory,
                "proxy_compute": proxy_compute,
                "percent_active_slices": final_active_gates * 100,
                "gate_entropy": final_entropy
            }
            
            results_path = append_to_results_table(results, output_dir)
            
            params = {k: v for k, v in model_params.items()} if model_name == "NdLinearGated" else {}
            metrics = {
                "accuracy": best_acc,
                "train_time": train_time,
                "peak_memory_mb": peak_memory,
                "percent_active_slices": final_active_gates * 100,
                "gate_entropy": final_entropy,
                "proxy_compute": proxy_compute
            }
            
            summary_path = save_summary(model_name + param_suffix, dataset_name, params, metrics, output_dir)
            
            experiment_data = {
                "experiment_name": exp_name,
                "model": model_name,
                "dataset": dataset_name,
                **params,
                "config_file": args.config,
                "output_dir": output_dir,
                "summary_yaml": summary_path,
                "results_row": results_path,
                "plots": {
                    "accuracy_curve": f"{output_dir}/plots/{model_name + param_suffix}_{dataset_name}_accuracy_curve.png",
                    "entropy_curve": f"{output_dir}/plots/{model_name + param_suffix}_{dataset_name}_entropy_curve.png" if model_name == "NdLinearGated" else None,
                    "active_gate_distribution": f"{output_dir}/plots/{model_name + param_suffix}_{dataset_name}_active_gate_distribution.png" if model_name == "NdLinearGated" else None,
                    "profiler_plot": f"{output_dir}/profiler_breakdown/{model_name + param_suffix}_{dataset_name}.png" if args.run_profiler and dataset_name == "cifar10" else None
                }
            }
            
            update_experiment_manifest(experiment_data, output_dir)
            
            if model_name == "NdLinearGated" and dataset_name == config["gate_transfer"]["source_dataset"]:
                source_model_for_transfer = model
            
            model_results.append({
                "name": model_name + param_suffix,
                "proxy_compute": proxy_compute,
                "accuracy": best_acc
            })
        
        proxy_computes = [result["proxy_compute"] for result in model_results]
        accuracies = [result["accuracy"] for result in model_results]
        model_names = [result["name"] for result in model_results]
        
        logger.info(f"Preparing to plot proxy compute vs accuracy for {dataset_name}")
        logger.info(f"Number of models: {len(model_results)}")
        
        if len(model_results) > 0:
            plot_proxy_compute_vs_accuracy(proxy_computes, accuracies, model_names, 
                                         "ablation", dataset_name, output_dir)
        else:
            logger.warning(f"No model results available for {dataset_name}, skipping plot")
    
    if "gate_transfer" in config and hasattr(locals(), 'source_model_for_transfer'):
        source_dataset = config["gate_transfer"]["source_dataset"]
        target_datasets = config["gate_transfer"]["target_datasets"]
        
        logger.info(f"\n=== Running gate transfer studies from {source_dataset} ===")
        
        for target_dataset in target_datasets:
            logger.info(f"Transferring gates from {source_dataset} to {target_dataset}...")
            
            target_info = dataset_dict[target_dataset]
            
            perform_gate_transfer(
                source_model_for_transfer,
                target_info["train_loader"],
                target_info["test_loader"],
                device,
                source_dataset,
                target_dataset,
                target_info["input_shape"],
                target_info["num_classes"],
                output_dir
            )
    
    logger.info("\nAll experiments completed.")
    logger.info(f"Results saved to {output_dir}/results_table.csv")
    logger.info(f"Experiment manifest saved to {output_dir}/experiment_manifest.yaml")

if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_multi_dataset_model(model, datasets, device='cuda', batch_size=64, 
                            learning_rate=0.001, num_epochs=50, save_path='best_model.pth'):
    """
    Train model on multiple individual datasets - OPTIMIZED VERSION
    
    Args:
        model: PyTorch model to train
        datasets: List of individual datasets
        device: Device to use for training
        batch_size: Batch size for DataLoader
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        save_path: Path to save the best model
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    losses = []
    
    print(f"Training on {len(datasets)} individual datasets...")
    
    # Combine all datasets for more efficient training
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Combined dataset size: {len(combined_dataset)} samples")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        # Progress bar for batches
        batch_iterator = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for point_cloud, wireframe in batch_iterator:
            point_cloud = point_cloud.to(device, non_blocking=True)
            wireframe = wireframe.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(point_cloud)
            loss = criterion(output, wireframe)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * point_cloud.size(0)
            total_samples += point_cloud.size(0)
            
            # Update progress bar
            batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
        
        # Print every 5 epochs instead of 10
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    print(f'Training completed. Best loss: {best_loss:.6f}')
    return model, losses

def evaluate_multi_datasets(model, datasets, device='cuda', batch_size=64):
    """
    Evaluate model on multiple individual datasets - OPTIMIZED VERSION
    
    Args:
        model: PyTorch model to evaluate
        datasets: List of individual datasets
        device: Device to use for evaluation
        batch_size: Batch size for DataLoader
    
    Returns:
        dict: Evaluation metrics
    """
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating on {len(datasets)} individual datasets...")
    
    # Combine all test datasets for efficient evaluation
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Combined test dataset size: {len(combined_dataset)} samples")
    
    with torch.no_grad():
        batch_iterator = tqdm(dataloader, desc='Evaluating', leave=False)
        
        for point_cloud, wireframe in batch_iterator:
            point_cloud = point_cloud.to(device, non_blocking=True)
            wireframe = wireframe.to(device, non_blocking=True)
            
            output = model(point_cloud)
            loss = criterion(output, wireframe)
            
            total_loss += loss.item() * point_cloud.size(0)
            total_samples += point_cloud.size(0)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(wireframe.cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    # Calculate RMSE
    if all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    else:
        rmse = 0
    
    metrics = {
        'loss': avg_loss,
        'rmse': rmse,
        'total_samples': total_samples,
        'num_datasets': len([d for d in datasets if len(d) > 0])
    }
    
    print(f'Evaluation completed:')
    print(f'  Average Loss: {avg_loss:.6f}')
    print(f'  RMSE: {rmse:.6f}')
    print(f'  Total Samples: {total_samples}')
    print(f'  Active Datasets: {metrics["num_datasets"]}')
    
    return metrics

def train_batch_model(model, dataloader, device='cuda', learning_rate=0.001, 
                     num_epochs=100, save_path='best_model.pth'):
    """
    Train model on batch dataset (legacy function for compatibility)
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        for point_cloud, wireframe in dataloader:
            point_cloud = point_cloud.to(device)
            wireframe = wireframe.to(device)
            
            optimizer.zero_grad()
            output = model(point_cloud)
            loss = criterion(output, wireframe)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * point_cloud.size(0)
            total_samples += point_cloud.size(0)
        
        avg_loss = epoch_loss / total_samples
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def evaluate_batch_datasets(model, dataloader, device='cuda'):
    """
    Evaluate model on batch dataset (legacy function for compatibility)
    """
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for point_cloud, wireframe in dataloader:
            point_cloud = point_cloud.to(device)
            wireframe = wireframe.to(device)
            
            output = model(point_cloud)
            loss = criterion(output, wireframe)
            
            total_loss += loss.item() * point_cloud.size(0)
            total_samples += point_cloud.size(0)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(wireframe.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'total_samples': total_samples
    }

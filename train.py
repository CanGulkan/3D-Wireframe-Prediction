import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss
from models.EdgePredictor import EdgePredictor
from models.PointNetEncoder import PointNetEncoder
from dataset.PCtoWFdataset import PCtoWFdataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_or_sample_pointcloud(point_cloud, target_size=1024):
    """Pad or sample point cloud to fixed size"""
    current_size = len(point_cloud)
    if current_size >= target_size:
        # Sample randomly
        indices = np.random.choice(current_size, target_size, replace=False)
        return point_cloud[indices]
    else:
        # Pad with zeros or repeat points
        pad_size = target_size - current_size
        padding = np.zeros((pad_size, point_cloud.shape[1]))
        return np.vstack([point_cloud, padding])


def pad_adjacency_matrix(adj_matrix, max_vertices):
    """Pad adjacency matrix to max_vertices x max_vertices"""
    current_size = adj_matrix.shape[0]
    if current_size >= max_vertices:
        return adj_matrix[:max_vertices, :max_vertices]
    else:
        # Create new matrix with zeros
        padded_adj = np.zeros((max_vertices, max_vertices))
        # Copy original matrix to top-left corner
        padded_adj[:current_size, :current_size] = adj_matrix
        return padded_adj


def pad_vertices(vertices, max_vertices):
    """Pad vertices to max_vertices with zeros"""
    current_vertices = len(vertices)
    if current_vertices >= max_vertices:
        return vertices[:max_vertices]  # Truncate if too many
    else:
        pad_size = max_vertices - current_vertices
        padding = np.zeros((pad_size, 3))
        return np.vstack([vertices, padding])


def create_adjacency_matrix_from_predictions(edge_probs, edge_indices, num_vertices, threshold=0.5):
    """Convert edge predictions to adjacency matrix"""
    batch_size = edge_probs.shape[0]
    adj_matrices = torch.zeros(batch_size, num_vertices, num_vertices)
    
    for batch_idx in range(batch_size):
        for edge_idx, (i, j) in enumerate(edge_indices):
            if edge_probs[batch_idx, edge_idx] > threshold:
                adj_matrices[batch_idx, i, j] = 1
                adj_matrices[batch_idx, j, i] = 1  # Symmetric
                
    return adj_matrices


def create_edge_labels_from_adjacency(adj_matrix, edge_indices):
    """Create edge labels tensor from adjacency matrix"""
    batch_size = 1  # Single example
    num_edges = len(edge_indices)
    edge_labels = torch.zeros(batch_size, num_edges)
    
    for edge_idx, (i, j) in enumerate(edge_indices):
        if adj_matrix[i, j] == 1:
            edge_labels[0, edge_idx] = 1
            
    return edge_labels




def train_batch_model(batch_data, num_epochs=1000, learning_rate=0.001):
    """Train model on batch processed multiple datasets"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Get batch data
    point_clouds = batch_data['point_clouds']
    vertices = batch_data['vertices']
    adjacency_matrices = batch_data['adjacency_matrices']
    scalers = batch_data['scalers']
    original_datasets = batch_data['original_datasets']
    
    num_datasets = len(point_clouds)
    max_vertices = vertices.shape[1]
    
    logger.info(f"Training on {num_datasets} datasets")
    logger.info(f"Using {max_vertices} vertices (max from training data)")
    
    model = PointCloudToWireframe(input_dim=8, num_vertices=max_vertices).to(device)
    
    # Convert to tensors
    point_clouds_tensor = torch.FloatTensor(point_clouds).to(device)
    target_vertices_tensor = torch.FloatTensor(vertices).to(device)
    
    # Create edge labels for each dataset
    all_edge_labels = []
    for adj_matrix in adjacency_matrices:
        edge_labels = create_edge_labels_from_adjacency(
            adj_matrix,
            [(i, j) for i in range(max_vertices) for j in range(i+1, max_vertices)]
        )
        all_edge_labels.append(edge_labels.squeeze(0))
    
    edge_labels_tensor = torch.FloatTensor(np.array(all_edge_labels)).to(device)
    
    criterion = WireframeLoss(vertex_weight=50.0, edge_weight=0.1)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 700, 1700, 4000], gamma=0.3)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_vertex_rmse = float('inf')
    best_model_state = None
    loss_history = []
    vertex_rmse_history = []
    patience = 500
    patience_counter = 0
    current_vertex_rmse = 999999

    logger.info(f"Starting batch training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Train on all datasets in each epoch
        for batch_idx in range(num_datasets):
            optimizer.zero_grad()
            
            # Get single sample
            pc_sample = point_clouds_tensor[batch_idx:batch_idx+1]
            vertices_sample = target_vertices_tensor[batch_idx:batch_idx+1]
            edges_sample = edge_labels_tensor[batch_idx:batch_idx+1]
            
            # Forward pass
            predictions = model(pc_sample)
            
            # Calculate loss
            targets = {
                'vertices': vertices_sample,
                'edge_labels': edges_sample
            }
            loss_dict = criterion(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / num_datasets
        
        # Track progress
        loss_history.append(avg_epoch_loss)
        
        # Calculate average vertex RMSE across all training samples
        with torch.no_grad():
            total_rmse = 0.0
            for batch_idx in range(num_datasets):
                pc_sample = point_clouds_tensor[batch_idx:batch_idx+1]
                vertices_sample = target_vertices_tensor[batch_idx:batch_idx+1]
                
                predictions = model(pc_sample)
                pred_vertices_np = predictions['vertices'].cpu().numpy()[0]
                target_vertices_np = vertices_sample.cpu().numpy()[0]
                
                # Convert back to original scale for RMSE calculation
                num_original_vertices = len(original_datasets[batch_idx].vertices)
                pred_vertices_orig = scalers[batch_idx].inverse_transform(pred_vertices_np[:num_original_vertices])
                target_vertices_orig = scalers[batch_idx].inverse_transform(target_vertices_np[:num_original_vertices])
                
                # Ensure same dimensions
                min_vertices = min(len(pred_vertices_orig), len(target_vertices_orig))
                pred_trimmed = pred_vertices_orig[:min_vertices]
                target_trimmed = target_vertices_orig[:min_vertices]
                
                rmse = np.sqrt(np.mean((pred_trimmed - target_trimmed) ** 2))
                total_rmse += rmse
                
            current_vertex_rmse = total_rmse / num_datasets
            vertex_rmse_history.append(current_vertex_rmse)
        
        # Early stopping based on vertex RMSE and save best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
        
        # Early stopping check
        if patience_counter >= patience and epoch > 100:
            logger.info(f"Early stopping at epoch {epoch}! Vertex RMSE hasn't improved for {patience} epochs")
            break

        # Log progress 
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Avg Loss: {avg_epoch_loss:.6f} | "
                       f"Avg Vertex RMSE: {current_vertex_rmse:.6f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                       f"Time: {elapsed_time:.1f}s")
            
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model state with Vertex RMSE: {best_vertex_rmse:.6f}")
    
    logger.info(f"Batch training completed! Best loss: {best_loss:.6f}")
    
    return model, loss_history


def evaluate_batch_datasets(model, batch_data, device, max_vertices):
    """Evaluate the trained model on batch processed test datasets"""
    model.eval()
    results = []
    
    # Get batch data
    point_clouds = batch_data['point_clouds']
    vertices = batch_data['vertices']
    adjacency_matrices = batch_data['adjacency_matrices']
    scalers = batch_data['scalers']
    original_datasets = batch_data['original_datasets']
    
    num_datasets = len(point_clouds)
    
    for i in range(num_datasets):
        logger.info(f"Evaluating on test dataset {i+1}")
        
        with torch.no_grad():
            # Prepare input
            point_cloud_tensor = torch.FloatTensor(point_clouds[i]).unsqueeze(0).to(device)
            
            # Forward pass
            predictions = model(point_cloud_tensor)
            
            # Get predictions (only valid vertices, not padded ones)
            num_original_vertices = len(original_datasets[i].vertices)
            pred_vertices = predictions['vertices'].cpu().numpy()[0][:num_original_vertices]
            pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
            edge_indices = predictions['edge_indices']
            
            # Convert back to original scale
            pred_vertices_original = scalers[i].inverse_transform(pred_vertices)
            true_vertices_original = original_datasets[i].vertices
            
            # Ensure same number of vertices for comparison
            min_vertices = min(len(pred_vertices_original), len(true_vertices_original))
            pred_vertices_trimmed = pred_vertices_original[:min_vertices]
            true_vertices_trimmed = true_vertices_original[:min_vertices]
            
            # Calculate metrics
            vertex_mse = np.mean((pred_vertices_trimmed - true_vertices_trimmed) ** 2)
            vertex_rmse = np.sqrt(vertex_mse)
            
            # Edge accuracy (threshold at 0.5) - only consider edges between original vertices
            pred_adj_matrix = create_adjacency_matrix_from_predictions(
                torch.FloatTensor(pred_edge_probs).unsqueeze(0),
                edge_indices,
                max_vertices,
                threshold=0.5
            )[0].numpy()
            
            # Truncate to original size for comparison
            pred_adj_matrix = pred_adj_matrix[:num_original_vertices, :num_original_vertices]
            true_adj_matrix = original_datasets[i].edge_adjacency_matrix
            
            # Ensure both matrices have the same size
            min_size = min(pred_adj_matrix.shape[0], true_adj_matrix.shape[0])
            pred_adj_matrix_resized = pred_adj_matrix[:min_size, :min_size]
            true_adj_matrix_resized = true_adj_matrix[:min_size, :min_size]
            
            edge_accuracy = np.mean((pred_adj_matrix_resized == true_adj_matrix_resized).astype(float))
            
            # Edge precision and recall
            true_edges = (true_adj_matrix_resized == 1)
            pred_edges = (pred_adj_matrix_resized == 1)
            
            tp = np.sum(true_edges & pred_edges)
            fp = np.sum(~true_edges & pred_edges)
            fn = np.sum(true_edges & ~pred_edges)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'dataset_index': i,
                'vertex_rmse': vertex_rmse,
                'edge_accuracy': edge_accuracy,
                'edge_precision': precision,
                'edge_recall': recall,
                'edge_f1_score': f1_score,
                'predicted_vertices': pred_vertices_original,
                'predicted_adjacency': pred_adj_matrix_resized,
                'edge_probabilities': pred_edge_probs
            }
            
            results.append(result)
            
            logger.info(f"Dataset {i+1} - Vertex RMSE: {vertex_rmse:.6f}, Edge Accuracy: {edge_accuracy:.6f}")
    
    return results


def train_multi_dataset_model(train_datasets, num_epochs=1000, learning_rate=0.001):
    """Train model on multiple datasets"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    logger.info(f"Training on {len(train_datasets)} datasets")
    
    # Use fixed number of vertices based on maximum in training data
    max_vertices = max(len(dataset.vertices) for dataset in train_datasets)
    logger.info(f"Using {max_vertices} vertices (max from training data)")
    model = PointCloudToWireframe(input_dim=8, num_vertices=max_vertices).to(device)
    
    
    # Prepare training data from all datasets
    all_point_clouds = []
    all_target_vertices = []
    all_edge_labels = []
    all_scalers = []
    all_edge_labels = []
    all_scalers = []
    
    for dataset in train_datasets:
        # Fixed point cloud size - sample or pad to 1024 points
        fixed_pc = pad_or_sample_pointcloud(dataset.normalized_point_cloud, target_size=1024)
        all_point_clouds.append(fixed_pc)
        
        # Pad vertices to max_vertices
        padded_vertices = pad_vertices(dataset.normalized_vertices, max_vertices)
        all_target_vertices.append(padded_vertices)
        
        # Pad adjacency matrix to max_vertices
        padded_adj_matrix = pad_adjacency_matrix(dataset.edge_adjacency_matrix, max_vertices)
        
        # Create edge labels for padded vertices
        edge_labels = create_edge_labels_from_adjacency(
            padded_adj_matrix,
            [(i, j) for i in range(max_vertices) for j in range(i+1, max_vertices)]
        )
        all_edge_labels.append(edge_labels.squeeze(0))  # Remove batch dimension
        all_scalers.append(dataset.spatial_scaler)
    
    # Convert to tensors
    point_clouds_tensor = torch.FloatTensor(np.array(all_point_clouds)).to(device)
    target_vertices_tensor = torch.FloatTensor(np.array(all_target_vertices)).to(device)
    edge_labels_tensor = torch.FloatTensor(np.array(all_edge_labels)).to(device)
    
    criterion = WireframeLoss(vertex_weight=50.0, edge_weight=0.1)  # Extreme vertex focus
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)  # No weight decay
    # More aggressive learning rate schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 700, 1700, 4000], gamma=0.3)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_vertex_rmse = float('inf')  # Initialize best vertex RMSE
    best_model_state = None  # Initialize best model state
    loss_history = []
    vertex_rmse_history = []
    patience = 500  # Increased patience for better convergence
    patience_counter = 0
    current_vertex_rmse = 999999

    logger.info(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Train on all datasets in each epoch
        for batch_idx in range(len(train_datasets)):
            optimizer.zero_grad()
            
            # Get single sample
            pc_sample = point_clouds_tensor[batch_idx:batch_idx+1]
            vertices_sample = target_vertices_tensor[batch_idx:batch_idx+1]
            edges_sample = edge_labels_tensor[batch_idx:batch_idx+1]
            
            # Forward pass
            predictions = model(pc_sample)
            
            # Calculate loss
            targets = {
                'vertices': vertices_sample,
                'edge_labels': edges_sample
            }
            loss_dict = criterion(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_datasets)
        
        # Track progress
        loss_history.append(avg_epoch_loss)
        
        # Calculate average vertex RMSE across all training samples
        with torch.no_grad():
            total_rmse = 0.0
            for batch_idx in range(len(train_datasets)):
                pc_sample = point_clouds_tensor[batch_idx:batch_idx+1]
                vertices_sample = target_vertices_tensor[batch_idx:batch_idx+1]
                
                predictions = model(pc_sample)
                pred_vertices_np = predictions['vertices'].cpu().numpy()[0]
                target_vertices_np = vertices_sample.cpu().numpy()[0]
                
                # Convert back to original scale for RMSE calculation
                num_original_vertices = len(train_datasets[batch_idx].vertices)
                pred_vertices_orig = all_scalers[batch_idx].inverse_transform(pred_vertices_np[:num_original_vertices])
                target_vertices_orig = all_scalers[batch_idx].inverse_transform(target_vertices_np[:num_original_vertices])
                
                # Ensure same dimensions
                min_vertices = min(len(pred_vertices_orig), len(target_vertices_orig))
                pred_trimmed = pred_vertices_orig[:min_vertices]
                target_trimmed = target_vertices_orig[:min_vertices]
                
                rmse = np.sqrt(np.mean((pred_trimmed - target_trimmed) ** 2))
                total_rmse += rmse
                
            current_vertex_rmse = total_rmse / len(train_datasets)
            vertex_rmse_history.append(current_vertex_rmse)
        
        # Early stopping based on vertex RMSE and save best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
        
        # Early stopping check
        if patience_counter >= patience and epoch > 100:
            logger.info(f"Early stopping at epoch {epoch}! Vertex RMSE hasn't improved for {patience} epochs")
            break

        # Log progress 
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Avg Loss: {avg_epoch_loss:.6f} | "
                       f"Avg Vertex RMSE: {current_vertex_rmse:.6f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                       f"Time: {elapsed_time:.1f}s")
            
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model state with Vertex RMSE: {best_vertex_rmse:.6f}")
    
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    
    return model, loss_history


def evaluate_multi_datasets(model, test_datasets, device, max_vertices):
    """Evaluate the trained model on multiple test datasets"""
    model.eval()
    results = []
    
    for i, dataset in enumerate(test_datasets):
        logger.info(f"Evaluating on test dataset {i+1}")
        
        with torch.no_grad():
            # Prepare input with same preprocessing as training
            fixed_pc = pad_or_sample_pointcloud(dataset.normalized_point_cloud, target_size=1024)
            point_cloud_tensor = torch.FloatTensor(fixed_pc).unsqueeze(0).to(device)
            
            # Forward pass
            predictions = model(point_cloud_tensor)
            
            # Get predictions (only valid vertices, not padded ones)
            num_original_vertices = len(dataset.vertices)
            pred_vertices = predictions['vertices'].cpu().numpy()[0][:num_original_vertices]
            pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
            edge_indices = predictions['edge_indices']
            
            # Convert back to original scale
            pred_vertices_original = dataset.spatial_scaler.inverse_transform(pred_vertices)
            true_vertices_original = dataset.vertices
            
            # Ensure same number of vertices for comparison
            min_vertices = min(len(pred_vertices_original), len(true_vertices_original))
            pred_vertices_trimmed = pred_vertices_original[:min_vertices]
            true_vertices_trimmed = true_vertices_original[:min_vertices]
            
            # Calculate metrics
            vertex_mse = np.mean((pred_vertices_trimmed - true_vertices_trimmed) ** 2)
            vertex_rmse = np.sqrt(vertex_mse)
            
            # Edge accuracy (threshold at 0.5) - only consider edges between original vertices
            pred_adj_matrix = create_adjacency_matrix_from_predictions(
                torch.FloatTensor(pred_edge_probs).unsqueeze(0),
                edge_indices,
                max_vertices,
                threshold=0.5
            )[0].numpy()
            
            # Truncate to original size for comparison
            pred_adj_matrix = pred_adj_matrix[:num_original_vertices, :num_original_vertices]
            true_adj_matrix = dataset.edge_adjacency_matrix
            
            # Ensure both matrices have the same size
            min_size = min(pred_adj_matrix.shape[0], true_adj_matrix.shape[0])
            pred_adj_matrix_resized = pred_adj_matrix[:min_size, :min_size]
            true_adj_matrix_resized = true_adj_matrix[:min_size, :min_size]
            
            edge_accuracy = np.mean((pred_adj_matrix_resized == true_adj_matrix_resized).astype(float))
            
            # Edge precision and recall
            true_edges = (true_adj_matrix_resized == 1)
            pred_edges = (pred_adj_matrix_resized == 1)
            
            tp = np.sum(true_edges & pred_edges)
            fp = np.sum(~true_edges & pred_edges)
            fn = np.sum(true_edges & ~pred_edges)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'dataset_index': i,
                'vertex_rmse': vertex_rmse,
                'edge_accuracy': edge_accuracy,
                'edge_precision': precision,
                'edge_recall': recall,
                'edge_f1_score': f1_score,
                'predicted_vertices': pred_vertices_original,
                'predicted_adjacency': pred_adj_matrix_resized,
                'edge_probabilities': pred_edge_probs
            }
            
            results.append(result)
            
            logger.info(f"Dataset {i+1} - Vertex RMSE: {vertex_rmse:.6f}, Edge Accuracy: {edge_accuracy:.6f}")
    
    return results


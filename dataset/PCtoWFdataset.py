import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCtoWFdataset:
    """
    A comprehensive dataset class to load and manage point cloud and wireframe file pairs
    from separate training and testing directories with batch processing capabilities.
    """
    def __init__(self, train_pc_dir, train_wf_dir, test_pc_dir, test_wf_dir):
        """
        Initializes the dataset handler.

        Args:
            train_pc_dir (str): Directory for training point cloud (.xyz) files.
            train_wf_dir (str): Directory for training wireframe (.obj) files.
            test_pc_dir (str): Directory for testing point cloud (.xyz) files.
            test_wf_dir (str): Directory for testing wireframe (.obj) files.
        """
        self.train_pc_dir = train_pc_dir
        self.train_wf_dir = train_wf_dir
        self.test_pc_dir = test_pc_dir
        self.test_wf_dir = test_wf_dir
        
        self.train_files = []
        self.test_files = []
        
        # Batch processing attributes
        self.train_datasets = []
        self.test_datasets = []
        self.max_vertices = 0
        self.max_points = 0
        
        self._load_train_data()
        self._load_test_data()

    def _load_file_pairs(self, pc_dir, wf_dir):
        """
        Loads and pairs the point cloud and wireframe files from given directories.
        """
        pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.xyz')))
        wf_files = sorted(glob.glob(os.path.join(wf_dir, '*.obj')))
        
        file_pairs = []
        wf_map = {os.path.basename(f).split('.')[0]: f for f in wf_files}
        
        for pc_file in pc_files:
            pc_basename = os.path.basename(pc_file).split('.')[0]
            if pc_basename in wf_map:
                file_pairs.append({
                    'pointcloud': pc_file,
                    'wireframe': wf_map[pc_basename]
                })
        return file_pairs

    def _load_train_data(self):
        """
        Loads the training data file pairs.
        """
        self.train_files = self._load_file_pairs(self.train_pc_dir, self.train_wf_dir)
        logger.info(f"Found {len(self.train_files)} training file pairs.")

    def _load_test_data(self):
        """
        Loads the testing data file pairs.
        """
        self.test_files = self._load_file_pairs(self.test_pc_dir, self.test_wf_dir)
        logger.info(f"Found {len(self.test_files)} testing file pairs.")

    def get_train_files(self):
        """
        Returns the list of file pairs for the training set.
        """
        return self.train_files

    def get_test_files(self):
        """
        Returns the list of file pairs for the testing set.
        """
        return self.test_files

    def load_all_training_data(self, max_files=10):
        """Load and preprocess training file pairs with limit"""
        files_to_load = self.train_files[:max_files] if max_files else self.train_files
        logger.info(f"Loading {len(files_to_load)} training file pairs (max: {max_files})...")
        
        self.train_datasets = []
        for i, file_pair in enumerate(files_to_load):
            logger.info(f"Processing training file pair {i+1}/{len(files_to_load)}")
            
            # Create individual dataset for this file pair
            individual_dataset = IndividualDataset(
                file_pair['pointcloud'], 
                file_pair['wireframe']
            )
            
            # Load and process data
            individual_dataset.load_point_cloud()
            individual_dataset.load_wireframe()
            individual_dataset.create_adjacency_matrix()
            individual_dataset.normalize_data()
            
            self.train_datasets.append(individual_dataset)
            
            # Track maximum sizes
            self.max_vertices = max(self.max_vertices, len(individual_dataset.vertices))
            self.max_points = max(self.max_points, len(individual_dataset.point_cloud))
        
        logger.info(f"Loaded {len(self.train_datasets)} training datasets")
        logger.info(f"Max vertices: {self.max_vertices}, Max points: {self.max_points}")
        
    def load_all_testing_data(self, max_files=4):
        """Load and preprocess testing file pairs with limit"""
        files_to_load = self.test_files[:max_files] if max_files else self.test_files
        logger.info(f"Loading {len(files_to_load)} testing file pairs (max: {max_files})...")
        
        self.test_datasets = []
        for i, file_pair in enumerate(files_to_load):
            logger.info(f"Processing testing file pair {i+1}/{len(files_to_load)}")
            
            # Create individual dataset for this file pair
            individual_dataset = IndividualDataset(
                file_pair['pointcloud'], 
                file_pair['wireframe']
            )
            
            # Load and process data
            individual_dataset.load_point_cloud()
            individual_dataset.load_wireframe()
            individual_dataset.create_adjacency_matrix()
            individual_dataset.normalize_data()
            
            self.test_datasets.append(individual_dataset)
        
        logger.info(f"Loaded {len(self.test_datasets)} testing datasets")
        
    def get_training_batch_data(self, target_points=1024):
        """Get all training datasets as batched tensors for training"""
        if not self.train_datasets:
            raise ValueError("Must call load_all_training_data() first")
            
        batch_point_clouds = []
        batch_vertices = []
        batch_adjacency_matrices = []
        batch_scalers = []
        
        for dataset in self.train_datasets:
            # Pad or sample point cloud to fixed size
            fixed_pc = self._pad_or_sample_pointcloud(dataset.normalized_point_cloud, target_points)
            batch_point_clouds.append(fixed_pc)
            
            # Pad vertices to max_vertices
            padded_vertices = self._pad_vertices(dataset.normalized_vertices, self.max_vertices)
            batch_vertices.append(padded_vertices)
            
            # Pad adjacency matrix
            padded_adj = self._pad_adjacency_matrix(dataset.edge_adjacency_matrix, self.max_vertices)
            batch_adjacency_matrices.append(padded_adj)
            
            batch_scalers.append(dataset.spatial_scaler)
        
        return {
            'point_clouds': np.array(batch_point_clouds),
            'vertices': np.array(batch_vertices),
            'adjacency_matrices': np.array(batch_adjacency_matrices),
            'scalers': batch_scalers,
            'original_datasets': self.train_datasets
        }
    
    def get_testing_batch_data(self, target_points=1024):
        """Get all testing datasets as batched tensors for evaluation"""
        if not self.test_datasets:
            raise ValueError("Must call load_all_testing_data() first")
            
        batch_point_clouds = []
        batch_vertices = []
        batch_adjacency_matrices = []
        batch_scalers = []
        
        for dataset in self.test_datasets:
            # Pad or sample point cloud to fixed size
            fixed_pc = self._pad_or_sample_pointcloud(dataset.normalized_point_cloud, target_points)
            batch_point_clouds.append(fixed_pc)
            
            # Pad vertices to max_vertices
            padded_vertices = self._pad_vertices(dataset.normalized_vertices, self.max_vertices)
            batch_vertices.append(padded_vertices)
            
            # Pad adjacency matrix
            padded_adj = self._pad_adjacency_matrix(dataset.edge_adjacency_matrix, self.max_vertices)
            batch_adjacency_matrices.append(padded_adj)
            
            batch_scalers.append(dataset.spatial_scaler)
        
        return {
            'point_clouds': np.array(batch_point_clouds),
            'vertices': np.array(batch_vertices),
            'adjacency_matrices': np.array(batch_adjacency_matrices),
            'scalers': batch_scalers,
            'original_datasets': self.test_datasets
        }
    
    def _pad_or_sample_pointcloud(self, point_cloud, target_size):
        """Pad or sample point cloud to fixed size"""
        current_size = len(point_cloud)
        if current_size >= target_size:
            # Sample randomly
            indices = np.random.choice(current_size, target_size, replace=False)
            return point_cloud[indices]
        else:
            # Pad with zeros
            pad_size = target_size - current_size
            padding = np.zeros((pad_size, point_cloud.shape[1]))
            return np.vstack([point_cloud, padding])
    
    def _pad_vertices(self, vertices, max_vertices):
        """Pad vertices to max_vertices with zeros"""
        current_vertices = len(vertices)
        if current_vertices >= max_vertices:
            return vertices[:max_vertices]
        else:
            pad_size = max_vertices - current_vertices
            padding = np.zeros((pad_size, 3))
            return np.vstack([vertices, padding])
    
    def _pad_adjacency_matrix(self, adj_matrix, max_vertices):
        """Pad adjacency matrix to max_vertices x max_vertices"""
        current_size = adj_matrix.shape[0]
        if current_size >= max_vertices:
            return adj_matrix[:max_vertices, :max_vertices]
        else:
            padded_adj = np.zeros((max_vertices, max_vertices))
            padded_adj[:current_size, :current_size] = adj_matrix
            return padded_adj

    def get_individual_training_datasets(self, max_files=10):
        """
        Her training dosya çiftini ayrı ayrı yükleyip individual dataset listesi döndürür
        """
        files_to_load = self.train_files[:max_files] if max_files else self.train_files
        logger.info(f"Creating {len(files_to_load)} individual training datasets...")
        
        individual_datasets = []
        
        for i, file_pair in enumerate(files_to_load):
            logger.info(f"Creating individual dataset {i+1}/{len(files_to_load)}: {file_pair['pointcloud']} + {file_pair['wireframe']}")
            
            # Her dosya çifti için ayrı bir IndividualDataset oluştur
            individual_dataset = IndividualDataset(
                file_pair['pointcloud'], 
                file_pair['wireframe']
            )
            
            # Veriyi yükle ve işle
            individual_dataset.load_point_cloud()
            individual_dataset.load_wireframe()
            individual_dataset.create_adjacency_matrix()
            individual_dataset.normalize_data()
            
            individual_datasets.append(individual_dataset)
            
            logger.info(f"Dataset {i+1} loaded: {len(individual_dataset.point_cloud)} points, {len(individual_dataset.vertices)} vertices")
        
        logger.info(f"Created {len(individual_datasets)} individual training datasets")
        return individual_datasets
    
    def get_individual_testing_datasets(self, max_files=4):
        """
        Her testing dosya çiftini ayrı ayrı yükleyip individual dataset listesi döndürür
        """
        files_to_load = self.test_files[:max_files] if max_files else self.test_files
        logger.info(f"Creating {len(files_to_load)} individual testing datasets...")
        
        individual_datasets = []
        
        for i, file_pair in enumerate(files_to_load):
            logger.info(f"Creating individual dataset {i+1}/{len(files_to_load)}: {file_pair['pointcloud']} + {file_pair['wireframe']}")
            
            # Her dosya çifti için ayrı bir IndividualDataset oluştur
            individual_dataset = IndividualDataset(
                file_pair['pointcloud'], 
                file_pair['wireframe']
            )
            
            # Veriyi yükle ve işle
            individual_dataset.load_point_cloud()
            individual_dataset.load_wireframe()
            individual_dataset.create_adjacency_matrix()
            individual_dataset.normalize_data()
            
            individual_datasets.append(individual_dataset)
            
            logger.info(f"Dataset {i+1} loaded: {len(individual_dataset.point_cloud)} points, {len(individual_dataset.vertices)} vertices")
        
        logger.info(f"Created {len(individual_datasets)} individual testing datasets")
        return individual_datasets


class IndividualDataset(Dataset):
    """Individual dataset class for a single point cloud and wireframe pair"""
    
    def __init__(self, xyz_file, obj_file):
        self.xyz_file = xyz_file
        self.obj_file = obj_file
        self.point_cloud = None
        self.vertices = None
        self.edges = None
        self.edge_adjacency_matrix = None
        
    def __len__(self):
        """Return length of 1 since this is a single file pair"""
        return 1
    
    def __getitem__(self, idx):
        """Return the point cloud and wireframe data"""
        if idx != 0:
            raise IndexError("IndividualDataset only contains one item")
        
        if self.normalized_point_cloud is None or self.normalized_vertices is None:
            raise ValueError("Data not loaded and normalized yet")
        
        # Convert to tensors
        point_cloud_tensor = torch.FloatTensor(self.normalized_point_cloud)
        
        # Pad vertices to max size (38)
        max_vertices = 38
        current_vertices = len(self.normalized_vertices)
        if current_vertices < max_vertices:
            # Pad with zeros
            padding = np.zeros((max_vertices - current_vertices, 3))
            padded_vertices = np.vstack([self.normalized_vertices, padding])
        else:
            # Take only first max_vertices
            padded_vertices = self.normalized_vertices[:max_vertices]
        
        vertices_tensor = torch.FloatTensor(padded_vertices)
        
        return point_cloud_tensor, vertices_tensor
        
    def load_point_cloud(self):
        """Load point cloud data from XYZ file"""
        logger.info(f"Loading point cloud from {self.xyz_file}")
        data = []
        
        with open(self.xyz_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # X Y Z R G B A Intensity
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b, a = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    intensity = float(parts[7])
                    data.append([x, y, z, r, g, b, a, intensity])
        
        self.point_cloud = np.array(data)
        logger.info(f"Loaded {len(self.point_cloud)} points")
        return self.point_cloud
    
    def load_wireframe(self):
        """Load wireframe data from OBJ file"""
        logger.info(f"Loading wireframe from {self.obj_file}")
        vertices = []
        edges = []
        
        with open(self.obj_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                    
                if parts[0] == 'v':  # Vertex
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                elif parts[0] == 'l':  # Line (edge)
                    # OBJ format uses 1-based indexing
                    v1, v2 = int(parts[1]) - 1, int(parts[2]) - 1
                    edges.append([v1, v2])
        
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        
        logger.info(f"Loaded {len(self.vertices)} vertices and {len(self.edges)} edges")
        return self.vertices, self.edges
    
    def create_adjacency_matrix(self):
        """Create adjacency matrix from edges"""
        if self.vertices is None or self.edges is None:
            raise ValueError("Must load wireframe data first")
            
        n_vertices = len(self.vertices)
        self.edge_adjacency_matrix = np.zeros((n_vertices, n_vertices))
        
        for edge in self.edges:
            v1, v2 = edge[0], edge[1]
            self.edge_adjacency_matrix[v1, v2] = 1
            self.edge_adjacency_matrix[v2, v1] = 1  # Undirected graph
            
        return self.edge_adjacency_matrix
    
    def normalize_data(self, target_points=1024):
        """Normalize point cloud and vertex coordinates"""
        if self.point_cloud is None:
            raise ValueError("Must load point cloud first")
            
        # Sample or pad point cloud to target size
        n_points = len(self.point_cloud)
        if n_points > target_points:
            # Downsample randomly
            indices = np.random.choice(n_points, target_points, replace=False)
            sampled_point_cloud = self.point_cloud[indices]
        elif n_points < target_points:
            # Upsample by repeating points
            indices = np.random.choice(n_points, target_points, replace=True)
            sampled_point_cloud = self.point_cloud[indices]
        else:
            sampled_point_cloud = self.point_cloud
            
        # Normalize spatial coordinates (X, Y, Z)
        spatial_coords = sampled_point_cloud[:, :3]
        self.spatial_scaler = StandardScaler()
        normalized_spatial = self.spatial_scaler.fit_transform(spatial_coords)
        
        # Normalize color values (R, G, B, A) to [0, 1]
        color_vals = sampled_point_cloud[:, 3:7] / 255.0
        
        # Normalize intensity
        intensity = sampled_point_cloud[:, 7:8]
        self.intensity_scaler = StandardScaler()
        normalized_intensity = self.intensity_scaler.fit_transform(intensity)
        
        # Combine normalized features
        self.normalized_point_cloud = np.hstack([
            normalized_spatial, color_vals, normalized_intensity
        ])
        
        # Normalize vertex coordinates using same spatial scaler
        if self.vertices is not None:
            self.normalized_vertices = self.spatial_scaler.transform(self.vertices)
        
        logger.info("Data normalization completed")
        return self.normalized_point_cloud
    
    def find_nearest_points_to_vertices(self, k=5):
        """Find k nearest points for each vertex in the wireframe"""
        if self.normalized_point_cloud is None or self.normalized_vertices is None:
            raise ValueError("Must normalize data first")
            
        # Use only spatial coordinates for nearest neighbor search
        point_spatial = self.normalized_point_cloud[:, :3]
        vertex_spatial = self.normalized_vertices[:, :3]
        
        # Find k nearest points for each vertex
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_spatial)
        distances, indices = nbrs.kneighbors(vertex_spatial)
        
        self.vertex_to_points_mapping = {
            'distances': distances,
            'indices': indices
        }
        
        logger.info(f"Found {k} nearest points for each of {len(self.normalized_vertices)} vertices")
        return distances, indices

if __name__ == '__main__':
    # Initialize the unified dataset with your actual data
    dataset = PCtoWFdataset(
        train_pc_dir='dataset/train_dataset/point_cloud',
        train_wf_dir='dataset/train_dataset/wireframe',
        test_pc_dir='dataset/test_dataset/point_cloud',
        test_wf_dir='dataset/test_dataset/wireframe'
    )
    
    # Example 1: Get file pairs (basic functionality)
    train_set = dataset.get_train_files()
    test_set = dataset.get_test_files()
    
    print("\n--- Training File Pairs ---")
    for i, pair in enumerate(train_set):
        print(f"{i+1}. Point Cloud: {pair['pointcloud']}, Wireframe: {pair['wireframe']}")
        
    print("\n--- Testing File Pairs ---")
    for i, pair in enumerate(test_set):
        print(f"{i+1}. Point Cloud: {pair['pointcloud']}, Wireframe: {pair['wireframe']}")

    # Example 2: Load all data and get batch data (advanced functionality)
    dataset.load_all_training_data()
    dataset.load_all_testing_data()
    
    train_batch_data = dataset.get_training_batch_data(target_points=1024)
    test_batch_data = dataset.get_testing_batch_data(target_points=1024)
    
    print(f"\n--- Batch Processing Results ---")
    print(f"Training batch shape: {train_batch_data['point_clouds'].shape}")
    print(f"Training vertices shape: {train_batch_data['vertices'].shape}")
    print(f"Training adjacency shape: {train_batch_data['adjacency_matrices'].shape}")
    print(f"Testing batch shape: {test_batch_data['point_clouds'].shape}")
    print(f"Max vertices found: {dataset.max_vertices}")
    print(f"Max points found: {dataset.max_points}")
    
    # Example 3: Legacy compatibility
    legacy_train_dataset = dataset.load_training_dataset()
    legacy_test_dataset = dataset.load_testing_dataset()
    
    print(f"\n--- Legacy Compatibility ---")
    print(f"Legacy training dataset created: {legacy_train_dataset is not None}")
    print(f"Legacy testing dataset created: {legacy_test_dataset is not None}")

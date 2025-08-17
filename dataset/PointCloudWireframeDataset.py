import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloudWireframeDataset:
    """Dataset class for loading and preprocessing multiple point cloud and wireframe data pairs"""
    
    def __init__(self, file_pairs):
        """
        Initialize with multiple file pairs from PCtoWFdataset
        
        Args:
            file_pairs: List of dictionaries with 'pointcloud' and 'wireframe' keys
        """
        if not isinstance(file_pairs, list) or len(file_pairs) == 0:
            raise ValueError("file_pairs must be a non-empty list of file pair dictionaries")
            
        self.file_pairs = file_pairs
        self.datasets = []  # Will store individual dataset objects
        self.max_vertices = 0
        self.max_points = 0
        
    def load_all_data(self):
        """Load and preprocess all file pairs"""
        logger.info(f"Loading {len(self.file_pairs)} file pairs...")
        
        for i, file_pair in enumerate(self.file_pairs):
            logger.info(f"Processing file pair {i+1}/{len(self.file_pairs)}")
            
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
            
            self.datasets.append(individual_dataset)
            
            # Track maximum sizes
            self.max_vertices = max(self.max_vertices, len(individual_dataset.vertices))
            self.max_points = max(self.max_points, len(individual_dataset.point_cloud))
        
        logger.info(f"Loaded {len(self.datasets)} datasets")
        logger.info(f"Max vertices: {self.max_vertices}, Max points: {self.max_points}")
        
    def get_batch_data(self, target_points=1024):
        """Get all datasets as batched tensors for training"""
        if not self.datasets:
            raise ValueError("Must call load_all_data() first")
            
        batch_point_clouds = []
        batch_vertices = []
        batch_adjacency_matrices = []
        batch_scalers = []
        
        for dataset in self.datasets:
            # Pad or sample point cloud to fixed size
            fixed_pc = self.pad_or_sample_pointcloud(dataset.normalized_point_cloud, target_points)
            batch_point_clouds.append(fixed_pc)
            
            # Pad vertices to max_vertices
            padded_vertices = self.pad_vertices(dataset.normalized_vertices, self.max_vertices)
            batch_vertices.append(padded_vertices)
            
            # Pad adjacency matrix
            padded_adj = self.pad_adjacency_matrix(dataset.edge_adjacency_matrix, self.max_vertices)
            batch_adjacency_matrices.append(padded_adj)
            
            batch_scalers.append(dataset.spatial_scaler)
        
        return {
            'point_clouds': np.array(batch_point_clouds),
            'vertices': np.array(batch_vertices),
            'adjacency_matrices': np.array(batch_adjacency_matrices),
            'scalers': batch_scalers,
            'original_datasets': self.datasets
        }
    
    def pad_or_sample_pointcloud(self, point_cloud, target_size):
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
    
    def pad_vertices(self, vertices, max_vertices):
        """Pad vertices to max_vertices with zeros"""
        current_vertices = len(vertices)
        if current_vertices >= max_vertices:
            return vertices[:max_vertices]
        else:
            pad_size = max_vertices - current_vertices
            padding = np.zeros((pad_size, 3))
            return np.vstack([vertices, padding])
    
    def pad_adjacency_matrix(self, adj_matrix, max_vertices):
        """Pad adjacency matrix to max_vertices x max_vertices"""
        current_size = adj_matrix.shape[0]
        if current_size >= max_vertices:
            return adj_matrix[:max_vertices, :max_vertices]
        else:
            padded_adj = np.zeros((max_vertices, max_vertices))
            padded_adj[:current_size, :current_size] = adj_matrix
            return padded_adj


class IndividualDataset:
    """Individual dataset class for a single point cloud and wireframe pair"""
    
    def __init__(self, xyz_file, obj_file):
        self.xyz_file = xyz_file
        self.obj_file = obj_file
        self.point_cloud = None
        self.vertices = None
        self.edges = None
        self.edge_adjacency_matrix = None
        
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
    
    def normalize_data(self):
        """Normalize point cloud and vertex coordinates"""
        if self.point_cloud is None:
            raise ValueError("Must load point cloud first")
            
        # Normalize spatial coordinates (X, Y, Z)
        spatial_coords = self.point_cloud[:, :3]
        self.spatial_scaler = StandardScaler()
        normalized_spatial = self.spatial_scaler.fit_transform(spatial_coords)
        
        # Normalize color values (R, G, B, A) to [0, 1]
        color_vals = self.point_cloud[:, 3:7] / 255.0
        
        # Normalize intensity
        intensity = self.point_cloud[:, 7:8]
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

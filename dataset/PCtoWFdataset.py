import os
import sys
import glob
from sklearn.model_selection import train_test_split

from .PointCloudWireframeDataset import PointCloudWireframeDataset


class PCtoWFdataset:
    """
    A dataset class to load and manage point cloud and wireframe file pairs
    from separate training and testing directories.
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
        print(f"Found {len(self.train_files)} training file pairs.")

    def _load_test_data(self):
        """
        Loads the testing data file pairs.
        """
        self.test_files = self._load_file_pairs(self.test_pc_dir, self.test_wf_dir)
        print(f"Found {len(self.test_files)} testing file pairs.")

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

    def load_training_dataset(self, **kwargs):
        """
        Creates and returns a PointCloudWireframeDataset for the training set.
        """
        if not self.train_files:
            print("No training files available.")
            return None
        
        return PointCloudWireframeDataset(self.train_files, **kwargs)

    def load_testing_dataset(self, **kwargs):
        """
        Creates and returns a PointCloudWireframeDataset for the testing set.
        """
        if not self.test_files:
            print("No testing files available.")
            return None
            
        return PointCloudWireframeDataset(self.test_files, **kwargs)

if __name__ == '__main__':
    # Create directories and sample data for testing
    os.makedirs('dataset/train_dataset/point_cloud', exist_ok=True)
    os.makedirs('dataset/train_dataset/wireframe', exist_ok=True)
    os.makedirs('dataset/test_dataset/point_cloud', exist_ok=True)
    os.makedirs('dataset/test_dataset/wireframe', exist_ok=True)

    # Create sample training data
    for i in range(3):
        with open(f'dataset/train_dataset/point_cloud/sample_{i}.xyz', 'w') as f:
            f.write("0 0 0 255 0 0 255 1.0\n1 1 1 0 255 0 255 0.8\n2 0 0 0 0 255 255 0.6\n")
        with open(f'dataset/train_dataset/wireframe/sample_{i}.obj', 'w') as f:
            f.write("v 0 0 0\nv 1 1 1\nv 2 0 0\nl 1 2\nl 2 3\n")

    # Create sample test data
    for i in range(2):
        with open(f'dataset/test_dataset/point_cloud/sample_{i}.xyz', 'w') as f:
            f.write("0 0 0 255 0 0 255 1.0\n1 1 1 0 255 0 255 0.8\n")
        with open(f'dataset/test_dataset/wireframe/sample_{i}.obj', 'w') as f:
            f.write("v 0 0 0\nv 1 1 1\nl 1 2\n")

    # Initialize the dataset
    dataset = PCtoWFdataset(
        train_pc_dir='dataset/train_dataset/point_cloud',
        train_wf_dir='dataset/train_dataset/wireframe',
        test_pc_dir='dataset/test_dataset/point_cloud',
        test_wf_dir='dataset/test_dataset/wireframe'
    )
    
    # Get the training and testing files
    train_set = dataset.get_train_files()
    test_set = dataset.get_test_files()
    
    print("\n--- Training Files ---")
    for pair in train_set:
        print(f"Point Cloud: {pair['pointcloud']}, Wireframe: {pair['wireframe']}")
        
    print("\n--- Testing Files ---")
    for pair in test_set:
        print(f"Point Cloud: {pair['pointcloud']}, Wireframe: {pair['wireframe']}")

    # Example of loading the actual datasets
    train_torch_dataset = dataset.load_training_dataset()
    test_torch_dataset = dataset.load_testing_dataset()

    print(f"\nLoaded training dataset")
    print(f"Loaded testing dataset")

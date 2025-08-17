import torch
from dataset.PCtoWFdataset import PCtoWFdataset
from src.train import evaluate_multi_datasets, train_multi_dataset_model


def main():
    
    # Load datasets
    print("Loading datasets...")
    dataset_handler = PCtoWFdataset(
        train_pc_dir='dataset/train_dataset/point_cloud',
        train_wf_dir='dataset/train_dataset/wireframe',
        test_pc_dir='dataset/test_dataset/point_cloud',
        test_wf_dir='dataset/test_dataset/wireframe'
    )
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n" + "="*60)
    print("STARTING INDIVIDUAL MULTI-TRAINING (10 files)")
    print("="*60)

    # Her training dosya çiftini ayrı ayrı yükle
    individual_train_datasets = dataset_handler.get_individual_training_datasets(max_files=10)
    
    # Model oluştur - Basit bir model kullanalım
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, input_points=1024, input_features=8, max_vertices=38):
            super(SimpleModel, self).__init__()
            self.max_vertices = max_vertices
            self.layers = nn.Sequential(
                nn.Linear(input_points * input_features, 512),  # Point cloud input
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, max_vertices * 3)  # Output: max_vertices * 3 coordinates
            )
        
        def forward(self, x):
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)  # Flatten
            return self.layers(x).view(batch_size, self.max_vertices, 3)  # Reshape to vertices
    
    model = SimpleModel(input_points=1024, input_features=8, max_vertices=38)
    
    # Multi training yap - her dataset ayrı ayrı işlenecek
    model, loss_history = train_multi_dataset_model(
        model,
        individual_train_datasets, 
        device=device,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=50
    )
    
    print("\n" + "="*60)
    print("EVALUATING ON INDIVIDUAL TEST DATASETS (4 files)")
    print("="*60)
    
    # Her test dosya çiftini ayrı ayrı yükle
    individual_test_datasets = dataset_handler.get_individual_testing_datasets(max_files=4)
    
    # Max vertices'i training datasetlerinden bul
    max_vertices = max(len(dataset.vertices) for dataset in individual_train_datasets)
    
    test_results = evaluate_multi_datasets(model, individual_test_datasets, device=device, batch_size=64)
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    print(f"Overall Test Loss: {test_results['loss']:.6f}")
    print(f"Overall Test RMSE: {test_results['rmse']:.6f}")
    print(f"Total Test Samples: {test_results['total_samples']}")
    print(f"Active Test Datasets: {test_results['num_datasets']}")
    print()

    print("Training completed successfully!")
    print(f"✅ Training: {len(individual_train_datasets)} datasets processed")
    print(f"✅ Testing: {len(individual_test_datasets)} datasets evaluated")
    print(f"✅ Final Training Loss: {loss_history[-1]:.6f}")
    print(f"✅ Final Test RMSE: {test_results['rmse']:.6f}")


if __name__ == "__main__":
    main()

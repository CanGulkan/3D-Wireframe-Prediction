import torch
from dataset.PCtoWFdataset import PCtoWFdataset
from train import evaluate_batch_datasets, train_batch_model


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
    print("STARTING BATCH TRAINING")
    print("="*60)

    # Load training dataset with multiple files
    train_dataset = dataset_handler.load_training_dataset()
    
    # Load and preprocess all training data at once
    train_dataset.load_all_data()
    
    # Get batched data for training
    batch_data = train_dataset.get_batch_data(target_points=1024)

    model, loss_history = train_batch_model(
        batch_data, 
        num_epochs=500, 
        learning_rate=0.001
    )
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST DATASET")
    print("="*60)
    
    # Get max vertices for evaluation
    max_vertices = train_dataset.max_vertices
    
    # Load and evaluate test dataset
    test_dataset = dataset_handler.load_testing_dataset()
    test_dataset.load_all_data()
    test_batch_data = test_dataset.get_batch_data(target_points=1024)
    
    test_results = evaluate_batch_datasets(model, test_batch_data, device, max_vertices)
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for result in test_results:
        print(f"Test Dataset {result['dataset_index']+1}:")
        print(f"  Vertex RMSE: {result['vertex_rmse']:.6f}")
        print(f"  Edge Accuracy: {result['edge_accuracy']:.6f}")
        print(f"  Edge Precision: {result['edge_precision']:.6f}")
        print(f"  Edge Recall: {result['edge_recall']:.6f}")
        print(f"  Edge F1-Score: {result['edge_f1_score']:.6f}")
        print()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DATASET = "hf_dataset_embeddings"
OUTPUT_DIR = "classifier_output"
EMBEDDING_DIM = 1024
HIDDEN_DIMS = [512, 256]  # Hidden layer sizes
DROPOUT = 0.3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralClassifier(nn.Module):
    """
    Neural network classifier for GatorTron embeddings.

    Architecture:
    - Input: [1024] embedding
    - Hidden layers with BatchNorm and Dropout
    - Output: Single logit for binary classification
    """

    def __init__(self, input_dim=1024, hidden_dims=[512, 256], dropout=0.3):
        super(NeuralClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)  # Output shape: [batch_size]

def load_data():
    """Load embeddings dataset and prepare for training."""
    print("Loading embeddings dataset...")
    dataset_dict = load_from_disk(INPUT_DATASET)

    train_data = dataset_dict['train']
    test_data = dataset_dict['test']

    # Extract embeddings and labels
    X_train = np.array(train_data['embedding'])
    y_train = np.array(train_data['label'], dtype=np.float32)

    X_test = np.array(test_data['embedding'])
    y_test = np.array(test_data['label'], dtype=np.float32)

    print(f"\nDataset statistics:")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Embedding dimension: {X_train.shape[1]}")

    # Class distribution
    train_pos = y_train.sum()
    train_neg = len(y_train) - train_pos
    test_pos = y_test.sum()
    test_neg = len(y_test) - test_pos

    print(f"\nTrain distribution: IP (True)={int(train_pos)}, OP (False)={int(train_neg)}")
    print(f"Test distribution: IP (True)={int(test_pos)}, OP (False)={int(test_neg)}")

    # Calculate pos_weight for imbalanced dataset
    pos_weight = train_neg / train_pos
    print(f"\nCalculated pos_weight: {pos_weight:.4f}")

    return X_train, y_train, X_test, y_test, pos_weight

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders."""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * batch_X.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    return avg_loss, accuracy, all_predictions, all_probabilities, all_labels

def plot_training_history(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Training curves saved to {output_dir}/training_history.png")

def plot_confusion_matrix(cm, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['OP (False)', 'IP (True)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.png")

def plot_pr_curve(precision, recall, pr_auc, output_dir):
    """Plot precision-recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ PR curve saved to {output_dir}/pr_curve.png")

def main():
    print("=" * 80)
    print("Neural Classifier Training for GatorTron Embeddings")
    print("=" * 80)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test, pos_weight = load_data()

    # Create dataloaders
    print(f"\nCreating DataLoaders (batch_size={BATCH_SIZE})...")
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, BATCH_SIZE
    )

    # Initialize model
    print(f"\nInitializing model on {DEVICE}...")
    model = NeuralClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with pos_weight for imbalanced data
    pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print(f"\nTraining configuration:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Pos_weight: {pos_weight:.4f}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Evaluate
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    val_loss, val_acc, predictions, probabilities, labels = evaluate(
        model, test_loader, criterion, DEVICE
    )

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)

    # Metrics
    print(f"\nTest Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.4f}")

    # ROC AUC
    roc_auc = roc_auc_score(labels, probabilities)
    print(f"ROC AUC: {roc_auc:.4f}")

    # PR AUC
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        labels, predictions,
        target_names=['OP (False)', 'IP (True)'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Save model
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
    print(f"✓ Model saved to {OUTPUT_DIR}/best_model.pt")

    # Save full model for inference
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': EMBEDDING_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': DROPOUT
        },
        'pos_weight': pos_weight,
        'training_config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': epoch + 1
        }
    }, f"{OUTPUT_DIR}/full_checkpoint.pt")
    print(f"✓ Full checkpoint saved to {OUTPUT_DIR}/full_checkpoint.pt")

    # Save metrics
    metrics = {
        'test_loss': float(val_loss),
        'test_accuracy': float(val_acc),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            labels, predictions,
            target_names=['OP (False)', 'IP (True)'],
            output_dict=True
        )
    }

    with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {OUTPUT_DIR}/metrics.json")

    # Save training history
    with open(f"{OUTPUT_DIR}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {OUTPUT_DIR}/training_history.json")

    # Save predictions
    np.savez(
        f"{OUTPUT_DIR}/predictions.npz",
        predictions=predictions,
        probabilities=probabilities,
        labels=labels
    )
    print(f"✓ Predictions saved to {OUTPUT_DIR}/predictions.npz")

    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, OUTPUT_DIR)
    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_pr_curve(precision, recall, pr_auc, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"  - best_model.pt: Model weights")
    print(f"  - full_checkpoint.pt: Complete checkpoint with config")
    print(f"  - metrics.json: Test metrics")
    print(f"  - training_history.json: Loss/accuracy per epoch")
    print(f"  - predictions.npz: Model predictions and probabilities")
    print(f"  - training_history.png: Training curves")
    print(f"  - confusion_matrix.png: Confusion matrix visualization")
    print(f"  - pr_curve.png: Precision-recall curve")

if __name__ == "__main__":
    main()

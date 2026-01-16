import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
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
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DATASET = "hf_dataset"
OUTPUT_DIR = "gatortron_finetuned"
MODEL_ID = "UFNLP/gatortron-base-2k"
MAX_LENGTH = 1900
CHUNK_OVERLAP = 100
BATCH_SIZE = 1  # Smaller batch size for fine-tuning
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 4 * 2 = 8
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
BASE_LR = 2e-5  # Learning rate for GatorTron base model
HEAD_LR = 1e-4  # Learning rate for classification head
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chunk_text(text, tokenizer, max_length=MAX_LENGTH, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks that fit within max_length tokens.
    Same as in generate_embeddings.py
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_length:
        return [text]

    chunks = []
    stride = max_length - overlap

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + max_length >= len(tokens):
            break

    return chunks

class GatorTronClassifier(nn.Module):
    """
    GatorTron with classification head for fine-tuning.

    Architecture:
    - GatorTron base model (trainable)
    - Classification head: Linear -> Dropout -> Linear
    - Hierarchical chunking: process chunks, mean-pool, classify
    """

    def __init__(self, model_id, dropout=0.3):
        super(GatorTronClassifier, self).__init__()

        # Load GatorTron base model
        self.gatortron = AutoModel.from_pretrained(model_id)
        self.hidden_size = self.gatortron.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for a single chunk.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size]
        """
        # Get GatorTron outputs
        outputs = self.gatortron(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Classification
        logits = self.classifier(cls_embedding).squeeze(1)  # [batch_size]

        return logits

class MedicalNotesDataset(Dataset):
    """
    Dataset that handles hierarchical chunking of medical notes.
    """

    def __init__(self, notes, labels, tokenizer, max_length=MAX_LENGTH):
        self.notes = notes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        note = self.notes[idx]
        label = self.labels[idx]

        # Chunk the note
        chunks = chunk_text(note, self.tokenizer, self.max_length)

        # Tokenize all chunks
        chunk_encodings = []
        for chunk in chunks:
            encoding = self.tokenizer(
                chunk,
                return_tensors="pt",
                max_length=self.max_length + 50,
                truncation=True,
                padding='max_length'
            )
            chunk_encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })

        return {
            'chunks': chunk_encodings,
            'label': torch.tensor(label, dtype=torch.float32),
            'num_chunks': len(chunk_encodings)
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable number of chunks.
    Processes all chunks from all documents in the batch.
    """
    all_input_ids = []
    all_attention_masks = []
    chunk_to_doc = []  # Maps chunk index to document index in batch
    labels = []

    for doc_idx, item in enumerate(batch):
        labels.append(item['label'])
        for chunk in item['chunks']:
            all_input_ids.append(chunk['input_ids'])
            all_attention_masks.append(chunk['attention_mask'])
            chunk_to_doc.append(doc_idx)

    return {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_attention_masks),
        'chunk_to_doc': torch.tensor(chunk_to_doc),
        'labels': torch.stack(labels),
        'batch_size': len(batch)
    }

def load_data(tokenizer):
    """Load dataset and prepare for training."""
    print("Loading dataset...")
    dataset_dict = load_from_disk(INPUT_DATASET)

    train_data = dataset_dict['train']
    test_data = dataset_dict['test']

    # Extract notes and labels
    train_notes = train_data['notes']
    train_labels = [label.upper() == "IP" for label in train_data['class']]

    test_notes = test_data['notes']
    test_labels = [label.upper() == "IP" for label in test_data['class']]

    print(f"\nDataset statistics:")
    print(f"Train samples: {len(train_notes)}")
    print(f"Test samples: {len(test_notes)}")

    # Class distribution
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos

    print(f"\nTrain distribution: IP (True)={train_pos}, OP (False)={train_neg}")
    print(f"Test distribution: IP (True)={test_pos}, OP (False)={test_neg}")

    # Calculate pos_weight
    pos_weight = train_neg / train_pos
    print(f"\nCalculated pos_weight: {pos_weight:.4f}")

    # Create datasets
    train_dataset = MedicalNotesDataset(train_notes, train_labels, tokenizer)
    test_dataset = MedicalNotesDataset(test_notes, test_labels, tokenizer)

    return train_dataset, test_dataset, pos_weight

def train_epoch(model, train_loader, criterion, optimizer, device, gradient_accumulation_steps):
    """Train for one epoch with hierarchical chunking."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        chunk_to_doc = batch['chunk_to_doc'].to(device)
        labels = batch['labels'].to(device)
        batch_size = batch['batch_size']

        # Forward pass through all chunks
        chunk_logits = model(input_ids, attention_mask)  # [total_chunks]

        # Mean-pool chunk logits for each document
        doc_logits = torch.zeros(batch_size, device=device)
        for doc_idx in range(batch_size):
            chunk_mask = (chunk_to_doc == doc_idx)
            doc_logits[doc_idx] = chunk_logits[chunk_mask].mean()

        # Calculate loss
        loss = criterion(doc_logits, labels)
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        predictions = (torch.sigmoid(doc_logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += batch_size

    # Final optimizer step if there are remaining gradients
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model with hierarchical chunking."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chunk_to_doc = batch['chunk_to_doc'].to(device)
            labels = batch['labels'].to(device)
            batch_size = batch['batch_size']

            # Forward pass through all chunks
            chunk_logits = model(input_ids, attention_mask)

            # Mean-pool chunk logits for each document
            doc_logits = torch.zeros(batch_size, device=device)
            for doc_idx in range(batch_size):
                chunk_mask = (chunk_to_doc == doc_idx)
                doc_logits[doc_idx] = chunk_logits[chunk_mask].mean()

            # Calculate loss
            loss = criterion(doc_logits, labels)
            total_loss += loss.item() * batch_size

            # Get predictions
            probabilities = torch.sigmoid(doc_logits)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
    print("GatorTron End-to-End Fine-tuning")
    print("=" * 80)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load data
    train_dataset, test_dataset, pos_weight = load_data(tokenizer)

    # Create dataloaders
    print(f"\nCreating DataLoaders (batch_size={BATCH_SIZE})...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    print(f"\nInitializing model on {DEVICE}...")
    model = GatorTronClassifier(MODEL_ID, dropout=0.3).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gatortron_params = sum(p.numel() for p in model.gatortron.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\nModel statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"GatorTron parameters: {gatortron_params:,}")
    print(f"Classifier head parameters: {classifier_params:,}")

    # Loss function
    pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer with differential learning rates
    optimizer = optim.AdamW([
        {'params': model.gatortron.parameters(), 'lr': BASE_LR},
        {'params': model.classifier.parameters(), 'lr': HEAD_LR}
    ], weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BASE_LR, HEAD_LR],
        total_steps=total_steps,
        pct_start=WARMUP_RATIO,
        anneal_strategy='cos'
    )

    print(f"\nTraining configuration:")
    print(f"  Base model LR: {BASE_LR}")
    print(f"  Classifier head LR: {HEAD_LR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
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
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, GRADIENT_ACCUMULATION_STEPS
        )

        # Evaluate
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"→ New best model (val_loss: {val_loss:.4f})")
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

    # Save full model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_id': MODEL_ID,
            'dropout': 0.3
        },
        'training_config': {
            'base_lr': BASE_LR,
            'head_lr': HEAD_LR,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'epochs': epoch + 1,
            'pos_weight': pos_weight
        }
    }, f"{OUTPUT_DIR}/best_model.pt")
    print(f"✓ Model saved to {OUTPUT_DIR}/best_model.pt")

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
    print("Fine-tuning Complete!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

import os
import torch
import pandas as pd
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Parameters
DATA_PATH = "training/gutenberg_dataset_chunks.csv"
MODEL_SAVE_PATH = "models/Trained_Roberta_new.pth"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
BALANCE_TYPE = 'fixed'
SAMPLE_SIZE = 200


# Device setup: Use MPS on Mac, cuda on Windows/Linux, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load Pretrained RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class AuthorDataset(Dataset):
    """Custom PyTorch Dataset for Author Classification."""
    
    def __init__(self, texts: pd.Series, labels: pd.Series):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label, dtype=torch.long)

def load_data(data_path: str, balance_type: str = BALANCE_TYPE, sample_size: int = None) -> tuple:
    """Loads the dataset and balances the number of samples per author before splitting into training and testing sets.

    Args:
        data_path (str): Path to dataset CSV file.
        balance_type (str): 
            - "min": Uses the minimum count for any author (undersampling).
            - "fixed": Uses a set sample size per author (defined by sample_size), 
                      only using replacement when needed.
        sample_size (int, optional): The number of samples per author if balance_type="fixed".

    Returns:
        tuple: Train/Test splits and author mapping.
    """
    logger.info("Loading dataset...")

    df = pd.read_csv(data_path).dropna()

    # Convert author names to numerical labels
    author_map = {author: idx for idx, author in enumerate(df["author"].unique())}
    df["author_label"] = df["author"].map(author_map)

    # Determine the minimum sample count per author
    min_samples_per_author = df["author"].value_counts().min()

    # Apply selected balancing method
    if balance_type == "min":
        logger.info(f"📊 Using minimum samples per author: {min_samples_per_author} (undersampling).")
        balanced_df = df.groupby("author").apply(lambda x: x.sample(min_samples_per_author, random_state=42)).reset_index(drop=True)

    elif balance_type == "fixed":
        if sample_size is None:
            raise ValueError("⚠️ 'sample_size' must be specified when using balance_type='fixed'.")

        logger.info(f"📊 Using {sample_size} samples per author, only replacing when necessary.")
        balanced_df = df.groupby("author").apply(
            lambda x: x.sample(n=sample_size, replace=(len(x) < sample_size), random_state=42)
        ).reset_index(drop=True)

    else:
        raise ValueError("⚠️ Invalid balance_type. Choose 'min' or 'fixed'.")

    # Save author map as JSON
    with open("assets/author_map.json", "w") as f:
        json.dump(author_map, f)
    logger.info("✅ Saved author_map.json")

    # Shuffle dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df["text"], balanced_df["author_label"], test_size=0.2, random_state=42
    )

    logger.info(f"Dataset loaded & balanced ({balance_type} mode). Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, author_map

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size: int) -> tuple:
    """Creates PyTorch DataLoaders for training and validation."""
    train_dataset = AuthorDataset(X_train, y_train)
    test_dataset = AuthorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Dataloaders created.")
    return train_loader, test_loader

def initialize_model(num_labels: int) -> torch.nn.Module:
    """Initializes the RoBERTa model for sequence classification."""
    logger.info("Initializing RoBERTa model...")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
    model.to(device)
    return model

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader) -> float:
    """Evaluates the model on the test set and returns the average validation loss."""
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(test_loader)
    logger.info(f"🔎 Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train_model(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    epochs: int, 
    save_path: str,
    patience: int = 5  # Early stopping patience
):
    """Trains the RoBERTa model with early stopping."""
    best_val_loss = float("inf")
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for batch_idx, batch in progress_bar:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"✅ Epoch {epoch+1} Completed, Avg Train Loss: {avg_train_loss:.4f}")

        # Validate the model
        val_loss = evaluate_model(model, test_loader)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
            logger.info(f"🎯 Best Model Saved! (Epoch {epoch+1}, Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"⚠️ No improvement for {epochs_no_improve} epochs.")

        # Early stopping condition
        if epochs_no_improve >= patience:
            logger.info(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Training complete! Best model saved as {save_path}")

def load_trained_model(model: torch.nn.Module, save_path: str):
    """Loads the best trained model."""
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("✅ Best model loaded for inference!")

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, author_map = load_data(data_path = DATA_PATH, sample_size = SAMPLE_SIZE)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, BATCH_SIZE)

    # Initialize model
    model = initialize_model(num_labels=len(author_map))

    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Start training
    logger.info("Starting training...")
    train_model(model, train_loader, test_loader, optimizer, epochs=EPOCHS, save_path=MODEL_SAVE_PATH)

    # Load best trained model for inference
    load_trained_model(model, MODEL_SAVE_PATH)
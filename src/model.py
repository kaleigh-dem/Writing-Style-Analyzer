import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, AUTHOR_MAP_PATH

# Load tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Device setup (Streamlit Cloud only supports CPU)
device = torch.device("cpu")

def load_trained_model():
    """Load the fine-tuned RoBERTa model from Hugging Face, ensuring compatibility."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)  # Move to CPU (ensures Streamlit Cloud compatibility)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Load model once to avoid reloading in each call
model = load_trained_model()

# Load author mapping
with open(AUTHOR_MAP_PATH, "r") as f:
    author_map = json.load(f)
label_to_author = {idx: author for author, idx in author_map.items()}

def classify_text(text):
    """Predicts the writing style percentages for an input passage."""
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    encoding = {key: val.to(device) for key, val in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    return {label_to_author[idx]: round(prob * 100, 2) for idx, prob in enumerate(probabilities)}
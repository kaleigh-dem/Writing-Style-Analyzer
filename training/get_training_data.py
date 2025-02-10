import os
import requests
import pandas as pd
import re
import nltk
from tqdm import tqdm

# Download NLTK tokenizer
nltk.download("punkt")

# Ensure the directory exists
books_dir = "training/gutenberg_books"
os.makedirs(books_dir, exist_ok=True)

# List of books to download from Project Gutenberg
books = {
    "Pride and Prejudice": {"author": "Jane Austen", "url": "https://www.gutenberg.org/files/1342/1342-0.txt"},
    "Moby Dick": {"author": "Herman Melville", "url": "https://www.gutenberg.org/files/2701/2701-0.txt"},
    "Great Expectations": {"author": "Charles Dickens", "url": "https://www.gutenberg.org/files/1400/1400-0.txt"},
    "The Picture of Dorian Gray": {"author": "Oscar Wilde", "url": "https://www.gutenberg.org/files/174/174-0.txt"},
    "Crime and Punishment": {"author": "Fyodor Dostoevsky", "url": "https://www.gutenberg.org/files/2554/2554-0.txt"},
    "The Adventures of Sherlock Holmes": {"author": "Arthur Conan Doyle", "url": "https://www.gutenberg.org/files/1661/1661-0.txt"},
    "The Call of the Wild": {"author": "Jack London", "url": "https://www.gutenberg.org/files/215/215-0.txt"},
    "Heart of Darkness": {"author": "Joseph Conrad", "url": "https://www.gutenberg.org/files/219/219-0.txt"},
    "Dracula": {"author": "Bram Stoker", "url": "https://www.gutenberg.org/files/345/345-0.txt"},
    "Frankenstein": {"author": "Mary Shelley", "url": "https://www.gutenberg.org/files/84/84-0.txt"},
    "The Scarlet Letter": {"author": "Nathaniel Hawthorne", "url": "https://www.gutenberg.org/files/25344/25344-0.txt"},
    "The Count of Monte Cristo": {"author": "Alexandre Dumas", "url": "https://www.gutenberg.org/files/1184/1184-0.txt"},
    "Les Misérables": {"author": "Victor Hugo", "url": "https://www.gutenberg.org/files/135/135-0.txt"},
    "Anna Karenina": {"author": "Leo Tolstoy", "url": "https://www.gutenberg.org/files/1399/1399-0.txt"},
    "Jane Eyre": {"author": "Charlotte Brontë", "url": "https://www.gutenberg.org/files/1260/1260-0.txt"},
    "Wuthering Heights": {"author": "Emily Brontë", "url": "https://www.gutenberg.org/files/768/768-0.txt"},
    "The Time Machine": {"author": "H.G. Wells", "url": "https://www.gutenberg.org/files/35/35-0.txt"},
    "The House of Mirth": {"author": "Edith Wharton", "url": "https://www.gutenberg.org/files/284/284-0.txt"},
    "Tess of the d'Urbervilles": {"author": "Thomas Hardy", "url": "https://www.gutenberg.org/files/110/110-0.txt"},
    "The Turn of the Screw": {"author": "Henry James", "url": "https://www.gutenberg.org/files/209/209-0.txt"},
    "Daisy Miller": {"author": "Henry James", "url": "https://www.gutenberg.org/files/208/208-0.txt"},
    "The Secret Agent: A Simple Tale": {"author": "Joseph Conrad", "url": "https://www.gutenberg.org/files/974/974-0.txt"},
    "The War of the Worlds": {"author": "H.G. Wells", "url": "https://www.gutenberg.org/files/36/36-0.txt"},
    "The Sea-Wolf": {"author": "Jack London", "url": "https://www.gutenberg.org/files/1074/1074-0.txt"},
}

# Function to clean Gutenberg text and remove headers, footers, and extra spaces
def clean_text(text):
    text = re.sub(r'\r\n', '\n', text)  # Normalize newlines
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = re.sub(r'_+', '', text)  # Remove Gutenberg's underline markers
    text = re.sub(r'(\*\*\* START OF[^*]+ \*\*\*)', '', text)  # Remove start header
    text = re.sub(r'(\*\*\* END OF[^*]+ \*\*\*)', '', text)  # Remove footer
    text = text.strip()  # Trim whitespace
    
    # Convert text to words and remove first and last 500 words
    words = text.split()
    if len(words) > 1000:  # Ensure text has more than 1000 words before slicing
        text = " ".join(words[500:-500])  
    return text

# Function to split text into chunks (e.g., 300-word sections)
def split_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Store book texts and metadata
data = []

for title, info in tqdm(books.items(), desc="Downloading Books"):
    response = requests.get(info["url"])
    
    if response.status_code == 200:
        text = response.text
        text = clean_text(text)
        
        # Save raw text to file
        with open(os.path.join(books_dir, f"{title}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        
        # Split into longer chunks (default: 300 words per chunk)
        chunks = split_into_chunks(text, chunk_size=300)
        
        # Store book text with labels
        for chunk in chunks:
            data.append({"text": chunk, "author": info["author"]})
    else:
        print(f"❌ Failed to download {title}")

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("training/gutenberg_dataset_chunks.csv", index=False)

# Print number of rows per author
author_counts = df["author"].value_counts()
print("\n📊 Number of rows per author:")
print(author_counts)

# Count the number of unique authors
num_authors = df["author"].nunique()
print(f"\n👤 Number of unique authors: {num_authors}")

print(f"\n✅ Dataset saved as 'gutenberg_dataset_chunks.csv' with {len(df)} samples!")
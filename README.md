# 📖 Writing Style Analyzer & Generator  

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)](https://streamlit.io/)
[![Roberta](https://img.shields.io/badge/NLP-RoBERTa-green)](https://huggingface.co/transformers/)

Analyze and Generate Text in the Style of Famous Authors  

🚀 A **Streamlit-powered web app** that leverages **RoBERTa** for **style analysis and text generation**, trained on a diverse set of authors.

---

## 🔍 Features

✔️ Classifies text based on author style.  
✔️ Generates new text mimicking the writing style of an author.  
✔️ Secure deployment with **Streamlit Cloud** and **OpenAI API protection**.  
✔️ Trained on **Gutenberg literary datasets**.

---

## 🌟 Demo

[🔗 Live Web App (Streamlit)](https://your-streamlit-app-link)

---

## 📂 Project Structure

```bash
Writing_Style_Analyzer/
│── .streamlit/                # Streamlit config (theme, secrets)
│── assets/                    # Stored JSON mappings, metadata
│── models/                    # Trained RoBERTa models
│── training/
│   │── Train_Roberta.py        # Model training script
│   │── gutenberg_dataset.csv   # Dataset for training
│── app.py                      # Streamlit Web App
│── style.py                    # Custom styling for UI
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Writing_Style_Analyzer.git
cd Writing_Style_Analyzer
```

### 2️⃣ Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys (For Streamlit Secrets)
Create a .streamlit/secrets.toml file:
```toml
[general]
OPENAI_API_KEY = "your-api-key"
PASSCODE = "your-passcode"
```

## 💡 Usage

### ▶ Run the Web App

```bash
streamlit run app.py
```

- Open localhost:8501 in your browser.
- Enter text and analyze its writing style. 
- Generate new text based on an author’s style.

## 🛠 Training Your Own Model

### 1️⃣ Prepare Dataset

Modify training/gutenberg_dataset_chunks.csv or use your own dataset with:

- Columns: text, author

### 2️⃣ Train RoBERTa Model

Run the training script:

```bash
python training/Train_Roberta.py
```

- Model will be saved in models/Trained_Roberta_new.pth.

### 3️⃣ Load Trained Model in app.py

Modify:

```bash
MODEL_PATH = "models/Trained_Roberta_new.pth"
```

## 🔒 Security & Deployment  

- API keys are **stored securely** in `.streamlit/secrets.toml` (Git-ignored).  
- Passcode protection prevents unauthorized access.  
- GitHub Secrets can be used for secure cloud deployment.  

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
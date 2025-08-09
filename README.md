# Rehla FAQ Chatbot

A chatbot that answers users' frequently asked questions (FAQ) about the **Rehla** service.  
It searches the FAQ database to find the most relevant answer to the user's query using **semantic search**.  
The chatbot also extracts keywords from the user's question for better context understanding.  

---

## Technologies Used

- **Backend Framework:** FastAPI  
- **Programming Language:** Python  
- **Semantic Embeddings:** SentenceTransformers  
- **Keyword Extraction:** KeyBERT  
- **Machine Learning Backend:** PyTorch  
- **Frontend:** HTML, CSS, JavaScript  
- **Data Storage:** JSON  

---

##  How to Run the Project

###  Install Dependencies
Run the following command in your terminal to install all dependencies:
```bash
pip install fastapi uvicorn sentence-transformers keybert torch
```

###  Run the Backend Server
```bash
uvicorn main:app --reload
```

Make sure the files `faq.json` and `main.py` are in the same directory.

---

Built by Futoon Abozaid

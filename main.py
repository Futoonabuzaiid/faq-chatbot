from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import json
import torch

# Initialize the FastAPI application
app = FastAPI()

# Enable CORS to allow frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load FAQ data from the JSON file
with open("faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Load the multilingual sentence embedding model (MiniLM)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the KeyBERT model using the embedding model
kw_model = KeyBERT(model)

# Extract all questions from the FAQ for similarity comparison
questions = [item["question"] for item in faq_data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Define the expected structure of the incoming request using Pydantic
class FAQRequest(BaseModel):
    message: str  # User's input message

# Define the POST endpoint for answering FAQ questions
@app.post("/faq")
def get_answer(req: FAQRequest):
    user_input = req.message  # Extract the message from request

    # Extract top 5 Arabic keywords from the user's input
    keywords = kw_model.extract_keywords(
        user_input,
        keyphrase_ngram_range=(1, 2),  # Extract unigrams and bigrams
        stop_words='arabic',           # Remove Arabic stopwords
        top_n=5                        # Limit to 5 keywords
    )
    extracted_keywords = [kw[0] for kw in keywords]  # Only get the keyword strings

    # Convert user's message into sentence embedding
    input_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity between input and each FAQ question
    cosine_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]
    best_score = float(torch.max(cosine_scores))  # Highest similarity score
    best_idx = int(torch.argmax(cosine_scores))   # Index of most similar question

    # If similarity is high, return the matched answer
    if best_score >= 0.7:
        answer = faq_data[best_idx]["answer"]
    else:
        # Otherwise, return a fallback message
        answer = "شكرًا لسؤالك. لم أجد إجابة مباشرة، سيتم التواصل معك من قبل فريق خدمة العملاء في أقرب وقت."

    # Return both the answer and extracted keywords
    return {
        "reply": answer,
        "keywords": extracted_keywords
    }

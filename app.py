# No keyword searching version
# Web Content Q&A Tool for Hugging Face Spaces
# Optimized for memory constraints (2GB RAM) and 24-hour timeline
# Features: Ingest up to 3 URLs, ask questions, get concise answers using DistilBERT with PyTorch

import gradio as gr
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
from huggingface_hub import hf_hub_download, HfFolder
from huggingface_hub.utils import configure_http_backend
import requests as hf_requests

# Configure Hugging Face Hub to use a custom session with increased timeout and retries
def create_custom_session():
    session = hf_requests.Session()
    # Increase timeout to 30 seconds (default is 10 seconds)
    adapter = hf_requests.adapters.HTTPAdapter(max_retries=3)  # Retry 3 times on failure
    session.mount("https://", adapter)
    session.timeout = 30  # Set timeout to 30 seconds
    return session

# Set the custom session for Hugging Face Hub
configure_http_backend(backend_factory=create_custom_session)

# Global variables for in-memory storage (reset on app restart)
corpus = []  # List of paragraphs from URLs
embeddings = None  # Precomputed embeddings for retrieval
sources_list = []  # Source URLs for each paragraph

# Load models at startup (memory: ~370MB total)
# Retrieval model: all-mpnet-base-v2 (~110MB, 768-dim embeddings)
retriever = SentenceTransformer('all-mpnet-base-v2')

# Load PyTorch model for QA
# Model: distilbert-base-uncased-distilled-squad (~260MB)
try:
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
except Exception as e:
    print(f"Error loading model: {str(e)}. Retrying with force_download=True...")
    # Force re-download in case of corrupted cache
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad", force_download=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad", force_download=True)

# Set model to evaluation mode
model.eval()

# Apply quantization to the model for faster inference on CPU
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Create the QA pipeline with PyTorch
qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="pt", device=-1)  # device=-1 for CPU

def ingest_urls(urls):
    """
    Ingest up to 3 URLs, scrape content, and compute embeddings.
    Limits: 100 paragraphs per URL to manage memory (~0.5MB embeddings total).
    """
    global corpus, embeddings, sources_list
    # Clear previous data
    corpus.clear()
    sources_list.clear()
    embeddings = None
    
    # Parse URLs from input (one per line, max 3)
    url_list = [url.strip() for url in urls.split("\n") if url.strip()][:3]
    if not url_list:
        return "Error: Please enter at least one valid URL."
    
    # Headers to mimic browser and avoid blocking
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Scrape each URL
    for url in url_list:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()  # Raise exception for bad status codes
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract content from <p> and <div> tags for broader coverage
            elements = soup.find_all(['p', 'div'])
            paragraph_count = 0
            for elem in elements:
                text = elem.get_text().strip()
                # Filter short or empty text
                if text and len(text) > 20 and paragraph_count < 100:
                    corpus.append(text)
                    sources_list.append(url)
                    paragraph_count += 1
            if paragraph_count == 0:
                return f"Warning: No usable content found at {url}."
        except Exception as e:
            return f"Error ingesting {url}: {str(e)}. Check URL and try again."
    
    # Compute embeddings if content was ingested
    if corpus:
        # Embeddings: ~3KB per paragraph, ~900KB for 300 paragraphs (768-dim)
        embeddings = retriever.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
        return f"Success: Ingested {len(corpus)} paragraphs from {len(set(url_list))} URLs."
    return "Error: No valid content ingested."

def answer_question(question):
    """
    Answer a question using retrieved context and DistilBERT QA (PyTorch).
    Retrieves top 3 paragraphs to improve answer accuracy.
    If total context exceeds 512 tokens (DistilBERT's max length), it will be truncated automatically.
    """
    global corpus, embeddings, sources_list
    if not corpus or embeddings is None:
        return "Error: Please ingest URLs first."
    
    # Encode question into embedding
    question_embedding = retriever.encode(question, convert_to_tensor=True)
    
    # Compute cosine similarity with stored embeddings
    cos_scores = util.cos_sim(question_embedding, embeddings)[0]
    top_k = min(1, len(corpus))  # Get top 3 paragraphs to improve accuracy
    top_indices = np.argsort(-cos_scores)[:top_k]
    
    # Retrieve context (top 3 paragraphs)
    contexts = [corpus[i] for i in top_indices]
    context = " ".join(contexts)  # Concatenate with space
    sources = [sources_list[i] for i in top_indices]
    
    # Extract answer with DistilBERT (PyTorch)
    with torch.no_grad():  # Disable gradient computation for faster inference
        result = qa_model(question=question, context=context)
    answer = result['answer']
    confidence = result['score']
    
    # Format response with answer, confidence, and sources
    sources_str = "\n".join(set(sources))  # Unique sources
    return f"Answer: {answer}\nConfidence: {confidence:.2f}\nSources:\n{sources_str}"

def clear_all():
    """Clear all inputs and outputs for a fresh start."""
    global corpus, embeddings, sources_list
    corpus.clear()
    embeddings = None
    sources_list.clear()
    return "", "", ""

# Gradio UI with minimal, user-friendly design
with gr.Blocks(title="Web Content Q&A Tool") as demo:
    gr.Markdown(
        """
        # Web Content Q&A Tool
        Enter up to 3 URLs (one per line), ingest their content, and ask questions.
        Answers are generated using only the ingested data. Note: Data resets on app restart.
        """
    )
    
    # URL input and ingestion
    with gr.Row():
        url_input = gr.Textbox(label="Enter URLs (one per line, max 3)", lines=3, placeholder="https://example.com")
        with gr.Column():
            ingest_btn = gr.Button("Ingest URLs")
            clear_btn = gr.Button("Clear All")
    ingest_output = gr.Textbox(label="Ingestion Status", interactive=False)
    
    # Question input and answer
    with gr.Row():
        question_input = gr.Textbox(label="Ask a question", placeholder="What is this about?")
        ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)
    
    # Bind functions to buttons
    ingest_btn.click(fn=ingest_urls, inputs=url_input, outputs=ingest_output)
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
    clear_btn.click(fn=clear_all, inputs=None, outputs=[url_input, ingest_output, answer_output])

# Launch the app (HF Spaces expects port 7860)
demo.launch(server_name="0.0.0.0", server_port=7860)

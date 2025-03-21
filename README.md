# Web Content Q&A Tool

A web-based Q&A tool that ingests content from up to 3 URLs and answers user questions using only the scraped data, built within a 24-hour timeline. The tool is deployed on Hugging Face Spaces and uses a retrieval-augmented approach with sentence embeddings and a question-answering model.

## Live Demo
You can access the live tool here: [https://huggingface.co/spaces/Shriharsh/Web_Content_QA](https://huggingface.co/spaces/Shriharsh/Web_Content_QA)

## Project Overview
This project was developed as part of an assignment to build a web-based Q&A tool with the following requirements:
- **Input**: Accept up to 3 URLs, scrape their content (text only).
- **Functionality**: Allow users to ask questions based solely on the ingested content, without relying on general knowledge.
- **Output**: Provide concise, accurate answers with sources.
- **Deliverables**: A live or hosted link (Hugging Face Spaces), source code on GitHub, and clear instructions to run locally.
- **Time Constraint**: Complete within 24 hours.

The tool uses a retrieval-augmented approach: it first retrieves relevant paragraphs from the ingested content using sentence embeddings, then extracts answers using a question-answering model. The final implementation is deployed on Hugging Face Spaces, with the source code available in this repository.

## Implementation Details
The tool is built using Python and leverages several libraries for web scraping, embeddings, and question-answering. Here’s a breakdown of the implementation:

### Tech Stack
- **Frontend**: Gradio for a minimal, user-friendly single-page web app.
- **Web Scraping**: `BeautifulSoup` to extract text from `<p>` and `<div>` tags of up to 3 URLs.
- **Retrieval**: `sentence-transformers` with `all-mpnet-base-v2` for generating embeddings and retrieving the top paragraph (`top_k=1`).
- **Question-Answering**: `transformers` with `distilbert-base-uncased-distilled-squad` for extracting answers from the retrieved context.
- **Framework**: PyTorch for model inference, with quantization to optimize for CPU.
- **Deployment**: Hosted on Hugging Face Spaces (free tier, CPU-only).

### Workflow
1. **Ingestion**:
   - Users input up to 3 URLs (one per line).
   - The tool scrapes content, limiting to 100 paragraphs per URL to manage memory (~900KB for embeddings of 300 paragraphs).
   - Paragraphs are embedded using `all-mpnet-base-v2` and stored in memory.
2. **Question-Answering**:
   - The user’s question is embedded using the same model.
   - Cosine similarity is computed to retrieve the top paragraph (`top_k=1`).
   - The paragraphs are concatenated and passed to the QA model.
   - The QA model extracts an answer, which is truncated to one line (max 100 characters) for conciseness.
3. **Output**:
   - The answer is displayed with its confidence score and source URLs.

### Key Features
- **Memory Optimization**: Quantized the QA model using `torch.quantization.quantize_dynamic` to reduce memory usage (~370MB total for models, ~1-1.2GB including embeddings).
- **Latency Management**: Used `top_k=1` to balance retrieval accuracy and speed, with inference times of ~90-150 seconds for 200 paragraphs.
- **User-Friendly UI**: Gradio interface with clear input fields for URLs and questions, and a concise output format (answer, confidence, sources).
- **Error Handling**: Robust handling for invalid URLs, empty content, and unanswerable questions.

## What We Tried and Discarded
During development, we explored several approaches to improve accuracy and performance, but discarded some due to time constraints, latency issues, or poor results:

1. **Keyword Search Fallback**:
   - **What We Tried**: Implemented a keyword search fallback when the QA model’s confidence was below 0.4, extracting keywords from the question (removing stop words) and retrieving paragraphs with the most keyword matches.
   - **Why Discarded**: The keyword search produced "horribly wrong answers with mismatching context" because it lacked semantic understanding, often retrieving irrelevant paragraphs (e.g., matching "machine" and "learning" in unrelated contexts).
   - **Lesson Learned**: Semantic embeddings are critical for retrieval in Q&A tasks, and simple keyword matching is insufficient.

2. **Confidence Threshold**:
   - **What We Tried**: Added a confidence threshold of 0.3 to reject low-confidence answers, returning "Unable to answer (confidence below 0.3)" to improve reliability.
   - **Why Discarded**: While this improved reliability by avoiding incorrect answers, I decided against it in the final version to ensure the tool always provides an answer, as my testing showed answers are always generated, even if sometimes incorrect at low confidence.
   - **Lesson Learned**: Balancing reliability and completeness is a trade-off; user feedback could guide whether to reintroduce this in the future.

3. **Advanced Models and Sentence Transformers**:
   - **What We Tried**: Experimented with `roberta-base-squad2` for QA and `multi-qa-mpnet-base-dot-v1` for retrieval, which improved accuracy but increased latency to ~120-190 seconds.
   - **Why Discarded**: The increased latency was too significant for the CPU environment on Hugging Face Spaces, especially with `top_k=1`.
   - **Lesson Learned**: More powerful models require better hardware (e.g., GPU) to maintain acceptable latency, which wasn’t feasible within the assignment’s constraints.

4. **Increasing `top_k`**:
   - **What We Tried**: Increased `top_k` to 3 to retrieve more paragraphs, hoping to improve answer accuracy by providing more context to the QA model.
   - **Why Discarded**: Retrieval took "a long time" (~500-670 seconds), and the accuracy gain was not proportional to the latency increase, so I reverted to `top_k=1`.
   - **Lesson Learned**: Retrieval latency scales with `top_k`, and a balance must be struck between context and performance on CPU.

## Engineering Decisions
Several key engineering decisions were made to meet the assignment’s requirements within the 24-hour timeline and resource constraints:

1. **Model Selection**:
   - Chose `all-mpnet-base-v2` (110MB) for retrieval due to its strong performance (STS score ~70%) and reasonable size, fitting within the 2GB memory limit.
   - Selected `distilbert-base-uncased-distilled-squad` (260MB) for QA because it’s lightweight, fast on CPU, and widely used for question-answering tasks.
   - Decision Rationale: These models balanced accuracy, memory usage, and inference speed on Hugging Face Spaces’ free tier (CPU-only).

2. **Memory Optimization**:
   - Quantized the QA model using `torch.quantization.quantize_dynamic` to reduce memory usage from ~260MB to ~150-200MB.
   - Limited ingestion to 100 paragraphs per URL (300 total) to keep embeddings memory usage at ~900KB (768 dimensions per paragraph).
   - Decision Rationale: Ensured the tool runs within Hugging Face Spaces’ 2GB RAM limit, with total usage at ~1-1.2GB.

3. **Latency Management**:
   - Set `top_k=1` to reduce retrieval time while still providing sufficient context for the QA model.
   - Used `torch.no_grad()` to disable gradient computation during inference, speeding up the QA model.
   - Decision Rationale: Kept inference time at ~90-150 seconds, acceptable for a CPU environment, though latency increases with more links (addressed below).

4. **Answer Formatting**:
   - Truncated answers to one line (max 100 characters) to ensure conciseness, as required by the assignment.
   - Ensured answers are at least one line by providing a default "No answer available" if the QA model fails to extract a meaningful answer.
   - Decision Rationale: Improved user experience by providing clear, concise responses, aligning with the assignment’s UI/UX criteria.

5. **Deployment on Hugging Face Spaces**:
   - Chose Hugging Face Spaces for hosting due to its free tier and seamless integration with Gradio and Hugging Face models.
   - Configured a custom session with increased timeout (30 seconds) and retries (3) to handle network issues during model downloads.
   - Decision Rationale: Met the assignment’s requirement for a live link while staying within budget (no costly APIs).

## Latency Issues
The tool faces inherent latency issues due to the CPU architecture and Hugging Face Spaces’ free tier constraints:
- **CPU Architecture**: Running on CPU (no GPU available) limits inference speed. Embedding 200 paragraphs with `all-mpnet-base-v2` takes ~10-20 seconds, and QA with `distilbert-base-uncased-distilled-squad` adds ~80-130 seconds, totaling ~90-150 seconds for answers.
- **Hugging Face Connection**: Network dependencies for downloading models and accessing the live space can introduce delays, especially during initial setup or if the space restarts.
- **Impact of More Links**: My testing shows that latency increases with more links (e.g., 3 URLs with 300 paragraphs vs. 1 URL with 100 paragraphs), as embedding time scales with the number of paragraphs. For 300 paragraphs, inference time can reach ~150 seconds, while 100 paragraphs is closer to ~90 seconds.

Despite these issues, my testing ensures that the tool always provides an answer, though accuracy varies at low confidence levels. The decision to use `top_k=1` mitigates some latency while maintaining reasonable accuracy.

## Testing Observations
- **Answer Availability**: The tool always generates an answer, even at low confidence (e.g., 0.04), which can sometimes be incorrect. This was a deliberate choice to prioritize completeness over reliability, though I explored a confidence threshold (discarded, as noted above).
- **Latency**: Inference time ranges from ~90-150 seconds for 100-300 paragraphs, increasing with more links due to the number of paragraphs embedded and processed.
- **Accuracy**: Answers are generally accurate when confidence is high (>0.5), but low-confidence answers can be incorrect, indicating retrieval or QA model limitations.
- **Memory**: Total usage is ~1-1.2GB, well within the 2GB limit, ensuring stability on Hugging Face Spaces.

## Future Improvements
With more time and resources, several enhancements could improve the tool’s performance, accuracy, and user experience:

1. **GPU-Powered Environments**:
   - Deploying on a GPU-powered environment (e.g., AWS, Google Colab Pro, or Hugging Face Spaces with paid tier) would significantly reduce latency. Embedding and QA inference could drop to ~20-40 seconds for 200 paragraphs, a 3-4x speedup.
   - GPU support would also enable the use of larger models without latency penalties.

2. **Advanced ML Models (Claude, GPT-4)**:
   - Replace the QA model with state-of-the-art models like Anthropic’s Claude or OpenAI’s GPT-4 for answer generation. These models excel at understanding context and generating accurate, natural-language answers, potentially eliminating the need for a separate retrieval step.
   - Challenges: Requires API access (costly), and answers must be constrained to ingested content, which may need prompt engineering (e.g., "Answer using only this context: [content]").

3. **State-of-the-Art Sentence Transformers**:
   - Upgrade to advanced sentence transformers like `all-roberta-large-v1` (355MB, STS score ~72%) or `multi-qa-MiniLM-L6-cos-v1` (optimized for QA, smaller footprint). These models could improve retrieval accuracy, ensuring more relevant paragraphs are passed to the QA model.
   - Challenges: Larger models increase memory usage and latency on CPU, but a GPU environment would mitigate this.

4. **Fine-Tuning**:
   - Fine-tune `all-mpnet-base-v2` or the QA model on a dataset of question-paragraph pairs (e.g., Wikipedia articles) to improve retrieval and answer extraction for this specific use case.
   - Challenges: Requires a dataset and significant compute time, which wasn’t feasible within the 24-hour timeline.

5. **Caching and Persistence**:
   - Implement caching for embeddings (e.g., using a database like SQLite) to avoid recomputing embeddings for the same URLs, reducing ingestion time on subsequent runs.
   - Challenges: Hugging Face Spaces’ free tier has limited persistent storage, so this would require a paid tier or alternative hosting.

6. **Confidence Threshold Reintroduction**:
   - Reintroduce a confidence threshold (e.g., 0.3) to reject low-confidence answers, improving reliability at the cost of completeness. User feedback could guide the threshold value.
   - Challenges: May reduce the tool’s ability to always provide an answer, which was a priority in this version.

7. **Cross-Encoder for Re-Ranking**:
   - Use a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the top 10 retrieved paragraphs, improving retrieval accuracy.
   - Challenges: Increases latency, but a GPU environment would make this feasible.

## Setup Instructions
### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   python app.py

## Running on Hugging Face Spaces
The tool is already deployed on Hugging Face Spaces. Visit the live link to test it directly: https://huggingface.co/spaces/Shriharsh/Web_Content_QA

## Acknowledgments
- Hugging Face for providing the free tier hosting and model hub.
- The sentence-transformers and transformers communities for their excellent documentation and pretrained models.
- The assignment for providing an opportunity to explore retrieval-augmented Q&A systems.

## Conclusion

This Web Content Q&A Tool meets the assignment’s requirements within the 24-hour timeline, providing a functional, user-friendly solution for ingesting URLs and answering questions. Despite challenges like latency on CPU and occasional low-confidence answers, the tool delivers concise, source-grounded responses. The development process involved critical engineering decisions to balance accuracy, latency, and memory usage, with several approaches tried and discarded to optimize performance. Future improvements, such as GPU deployment, advanced models, and fine-tuning, could further enhance the tool’s capabilities. I hope this project demonstrates my ability to tackle complex problems under tight constraints while maintaining a focus on user experience and code quality.

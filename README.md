
# AI-Powered RAG System for Legal & Medical Document Q&A

An AI system that uses Retrieval-Augmented Generation (RAG) to answer questions from Legal and Medical documents using semantic search and LLM reasoning.

---

## Problem Statement

Organizations store large volumes of legal contracts and medical reports. Manually searching through these documents is slow and inefficient.

Traditional keyword search cannot understand the meaning of questions.

This project builds an **AI-powered question answering system** that retrieves relevant document sections and generates answers using LLMs.

---

## System Architecture

User Question  
↓  
Query Embedding  
↓  
Vector Search (FAISS)  
↓  
Relevant Document Chunks  
↓  
Groq LLM (LLaMA 3)  
↓  
Generated Answer  

Pipeline:

PDF → Text Extraction → Chunking → Embeddings → FAISS Index → Retrieval → LLM Answer → Evaluation

---

## Tech Stack

| Technology | Version |
|-----------|--------|
Python | 3.10 |
LangChain | 0.2+ |
Sentence Transformers | all-MiniLM-L6-v2 |
Vector Database | FAISS |
LLM | Groq LLaMA-3 |
Evaluation | RAGAS |
Frontend | Streamlit |
Visualization | Matplotlib |

---

## Project Structure

```
AI-RAG-System
│
├── test_questions.json
│
├── logs
│   └── app.log
│
├── notebook
│   ├── Data_loder.ipynb
│   └── RAG_Legal_Medical_Document_QA.ipynb
│
├── output
│   │
│   ├── eval_results
│   │   ├── ragas_scores_chart.png
│   │   ├── ragas_scores.json
│   │   └── retrieval_sample_output.json
│   │
│   ├── faiss_indexes
│   │   ├── combined_index_meta.json
│   │   ├── combined_index.faiss
│   │   ├── legal_index_meta.json
│   │   ├── legal_index.faiss
│   │   ├── medical_index_meta.json
│   │   └── medical_index.faiss
│   │
│   ├── chunks_300.json
│   ├── chunks_500.json
│   ├── parsed_docs.json
│   └── rag_answers.json
│
├── src
│   ├── ingestion.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── llm_chain.py
│   ├── evaluator.py
│   └── utils.py
│
├── app
│   └── app.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1 Clone Repository

```
git clone https://github.com/your-username/AI-RAG-System.git
cd AI-RAG-System
```

### 2 Install Dependencies

```
pip install -r requirements.txt
```

### 3 Create Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 4 Run Streamlit Application

```
streamlit run app/app.py
```

---

## How to Get Groq API Key (Free)

1. Visit https://console.groq.com/keys  
2. Create a free account  
3. Generate an API key  
4. Add it to `.env`

---

## Sample Question & Answer

Question:

What is a contract?

Answer:

A contract is an agreement enforceable by law between two or more parties.

Source:

contract.pdf — Page 6

---

## RAGAS Evaluation Results

Evaluation metrics used:

• Faithfulness  
• Answer Relevancy  

Example:

Faithfulness: 0.60

This indicates the generated answer is moderately grounded in the retrieved document context.

Evaluation results are saved in:

```
output/eval_results/
```
Files include:

- ragas_scores.json
- ragas_scores_chart.png

---

## Challenges Faced

### PDF Parsing

Legal and medical documents contain complex formatting.

Solution:

Used **PyMuPDF** for reliable text extraction.

### Chunking Strategy

Large chunks reduce retrieval quality.

Solution:

Used **RecursiveCharacterTextSplitter** with overlap.

### API Limitations

Groq API restricts multiple responses.

Solution:

Adjusted evaluation pipeline and prompt structure.

---

## Future Improvements

Possible improvements:

• OCR support for scanned PDFs  
• Multi-document comparison  
• Hybrid retrieval (BM25 + vector search)  
• Conversation memory  
• Multi-language support  

---

## Deployment

Streamlit App:

(Add your Streamlit Cloud link here)

---

## Author

Sayali Moon  
AI / Data Science Project

---

## License

This project is open-source and available under the MIT License.

## Task Summary

| Task |     Task Name       | Description                                                                    | Deliverable |
| **T1** | PDF Ingestion | PyMuPDF parsing, page extraction, document type detection, metadata tagging         | `ingestion.py` |
| **T2** | Text Chunking | RecursiveCharacterTextSplitter, domain-aware chunking, overlap tuning               | `chunker.py` |
| **T3** | Embedding + FAISS | SBERT embeddings, FAISS index build/save/load, metadata store                   | `embedder.py` |
| **T4** | Retrieval Logic | Top-k similarity search, MMR diversification, confidence threshold,
domain filtering                                                                                                | `retriever.py` |
| **T5** | LLM + RAG Chain | Groq API integration, LangChain RetrievalQA pipeline, prompt templates, 
source citation, hallucination guard                                                                            | `llm_chain.py` |
| **T6** | RAGAS Evaluation | Faithfulness, answer relevancy, context recall, context precision 
metrics with visualization                                                                                      | `evaluator.py` |
| **T7** | Streamlit App | PDF upload interface, chat UI, source panel, confidence indicator, deployment           | `app.py` |
| **T8** | Error Handling | API retry logic, invalid input handling, logging system, environment variable
validation                                                                                                        | `utils.py` |
| **T9** | Documentation | Project README, architecture diagram, setup guide, GitHub repository documentation | `README.md` |


Project Title
Problem Statement
System Architecture
Tech Stack
Project Structure
Task Summary  ← yaha
Setup Instructions
Groq API Key Guide
Sample Outputs
RAGAS Evaluation
Challenges
Future Improvements


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
 . context_recall
 . context_precision
Example:


  "faithfulness": 0.7222222222222222,
  "answer_relevancy": 0.7975466680241915,
  "context_recall": 0.8333333333333334,
  "context_precision": 0.89999999993


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

Local URL: http://localhost:8501
Network URL: http://192.168.1.12:8501

---

## Author

Sayali Moon  
AI / Data Science Project

---

## License

This project is open-source and available under the MIT License.

## Task Summary
## 📋 Task Summary

The project is implemented in modular tasks to ensure scalability and maintainability.

| Task | Task Name | Description | Deliverable |
|-----|-----------|-------------|-------------|
| **T1** | PDF Ingestion | Extract text from PDFs using PyMuPDF, detect document type, and attach metadata such as page number and source file | `ingestion.py` |
| **T2** | Text Chunking | Split documents using RecursiveCharacterTextSplitter with optimized chunk size and overlap | `chunker.py` |
| **T3** | Embedding + FAISS | Generate semantic embeddings using SBERT and store them in FAISS vector index | `embedder.py` |
| **T4** | Retrieval Logic | Retrieve top-k relevant chunks using similarity search and apply filtering rules | `retriever.py` |
| **T5** | LLM + RAG Chain | Integrate Groq LLM with LangChain RetrievalQA pipeline and add prompt templates and citations | `llm_chain.py` |
| **T6** | RAGAS Evaluation | Evaluate system using Faithfulness and Answer Relevancy metrics and visualize results | `evaluator.py` |
| **T7** | Streamlit Application | Interactive web interface for document upload and question answering | `app.py` |
| **T8** | Error Handling | Implement API retry logic, logging, input validation and environment variable checks | `utils.py` |
| **T9** | Documentation | Project documentation including architecture diagram, setup instructions, and usage guide | `README.md` |


# AI-Powered RAG System for Legal & Medical Documents

## Problem Statement

## System Architecture

## Tech Stack

## Project Structure

## Task Summary

## Setup Instructions

## Getting Groq API Key

## Sample Q&A Outputs

## RAGAS Evaluation Results

## Challenges Faced

## Future Improvements

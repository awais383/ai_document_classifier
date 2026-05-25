# Document AI Pipeline
A fully local AI system for document classification, data extraction, and semantic search.
No paid APIs. No internet required after setup.

---

## What It Does
| Step | Description |
|------|-------------|
| 1. Ingest | Reads all PDF and TXT files from a folder |
| 2. Classify | Labels each document as Invoice / Resume / Utility Bill / Other / Unclassifiable |
| 3. Extract | Pulls structured fields from each document type |
| 4. Search | Lets you search documents by meaning using semantic embeddings |

---

## Libraries Used & Why

| Library | Purpose | Why Chosen |
|---------|---------|------------|
| `pdfminer.six` | Extract text from PDFs | Pure Python, handles complex layouts |
| `scikit-learn` | TF-IDF vectorizer (utility) | Lightweight, no heavy models needed |
| `sentence-transformers` | Text embeddings for search | Best open-source semantic model, offline |
| `faiss-cpu` | Fast similarity search index | Facebook's library, CPU-only, production-grade |
| `numpy` | Vector operations | Required by FAISS and embeddings |

---

## Installation

```bash
# 1. Clone or unzip the project
cd classifier

# 2. (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pdfminer.six sentence-transformers faiss-cpu scikit-learn numpy
```

---

## How to Run

### Step 1 — Add your documents
```
classifier/
└── documents/
    ├── invoice_1.pdf
    ├── resume_john.pdf
    ├── electricity_bill.pdf
    └── other_doc.txt
```

### Step 2 — Run the pipeline
```bash
# Basic run — classify + extract → output.json
python main.py --folder ./documents

# With interactive semantic search
python main.py --folder ./documents --search

# Single search query
python main.py --folder ./documents --query "payments due in January"
```

---

## Output Format

`output.json` is generated automatically:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-1234",
    "date": "2025-01-01",
    "company": "ACME Ltd.",
    "total_amount": 350.50
  },
  "resume_john.pdf": {
    "class": "Resume",
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "123-456-7890",
    "experience_years": 5
  },
  "electricity_bill.pdf": {
    "class": "Utility Bill",
    "account_number": "ACC-9876",
    "date": "2025-01-15",
    "usage_kwh": 312.5,
    "amount_due": 45.80
  }
}
```

---

## Classification Method

**Keyword Scoring (No training data needed)**
- Each document type has a set of domain-specific keywords
- Text is scored against each category
- Category with highest weighted score wins
- Threshold-based fallback to "Other" or "Unclassifiable"

## Extraction Method

**Regex-based pattern matching**
- Each field uses multiple regex patterns (handles variations in formatting)
- Falls back gracefully — returns `null` if not found
- Covers common date formats, currency symbols, phone formats

## Search Method

**SentenceTransformers + FAISS**
- Documents encoded into 384-dimensional semantic vectors
- Query encoded the same way
- FAISS finds nearest vectors by cosine similarity
- Returns top-3 most semantically relevant documents

---

## Project Structure

```
classifier/
├── main.py          # Entry point — orchestrates full pipeline
├── classifier.py    # Document classification logic
├── extractor.py     # Field extraction per document type
├── search.py        # Semantic search engine
├── output.json      # Generated after running
├── README.md        # This file
└── documents/       # Put your PDF/TXT files here
```

---

## Requirements

- Python 3.8+
- ~500MB disk space (for sentence-transformers model download)
- No GPU required
- No internet required after initial model download
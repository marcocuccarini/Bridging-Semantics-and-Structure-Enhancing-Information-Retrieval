# 🏛️ Bridging Semantics and Structure: Enhancing Information Retrieval with Knowledge Graphs in low resources context

This project performs semantic ranking and exploration over a knowledge graph of historical passages from three Italian cities: **Florence**, **Rome**, and **Venice**. It leverages multiple semantic models and graph traversal techniques to answer questions based on city-specific datasets.

---

## 📁 Directory Overview

```
├── main.py                 # Main script to run the experiment
├── requirements.txt        # Required Python libraries
├── KG/                     # Knowledge graphs of passages for each city
├── KG_entities/            # Knowledge graphs of entities extracted form the text for each city.
                            In combiantion with the dataset of passages can be used for the KG of Passages construction.  
├── dataset/                # Questions and corresponding passages
├── Classes/                # Core class definitions
├── results/                # Output from semantic and graph-based ranking

```

---

## 🚀 Getting Started

### Requirements

- Python 3.x
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Experiment

```bash
python3 main.py
```

This will execute all experiments and print results to the console.

---

## 🧠 Knowledge Graph (`KG/`)

Each JSON entry defines a relationship between two passages:

```json
{
  "from_id": 0,
  "from_text": "Florence Cathedral (Italian: Duomo di Firenze), formally the ---",
  "to_id": 34,
  "to_text": "The church is particularly notable for its 44 stained glass windows...",
  "value": 1
}
```

> Note: Relationships are treated as undirected, though directional fields (`from_id`, `to_id`) are preserved for clarity. This design allows for future improvements.

---

## ❓ Questions Dataset (`dataset/`)

Each city has a set of questions with corresponding passages:

```json
{
  "florence": {
    "0": {
      "question": "What architectural styles were used to design Florence Cathedral?",
      "passage": "Florence Cathedral (Italian: Duomo di Firenze), formally the Cathedral of Sai..."
    },
    ...
  }
}
```

---

## 🧪 What Happens During `main.py`

### 1. Semantic Ranking

For each question, the following models rank all passages by similarity:

- `Sentence-BERT (all-MiniLM-L6-v2)`
- `Sentence-BERT (all-mpnet-base-v2)`
- `SPADE` (all-MiniLM-L6-v2)
- `SPADE` (all-mpnet-base-v2)
- `BM25`

Each produces a ranked list of passages per city.

### 2. Graph-Based Question Expansion

- Selects top N most similar passages ("heads") per question.
- Explores related questions up to a configurable depth using the knowledge graph.
- Filters out:
  - Unrelated questions.
  - Questions linked to isolated passages (i.e., those with no graph connections).

---

## 📊 Output

Printed to the console:

- Ranked results for each model.
- Evaluation of semantic rankings.

  - Note that in the evaluation file output the level is counted from 0 to k-1; in the paper it is from 1 to k.


---

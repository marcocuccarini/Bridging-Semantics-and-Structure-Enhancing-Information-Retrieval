import os
import json
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

import json
import csv
import json
import os


import os
import csv

def get_correct_passage_ranks_with_text(results):
    """Return the rank of the correct passage for baseline or any ranking."""
    ranks = {}
    for qid, data in results.items():
        correct_passage_id = qid
        correct_rank = None
        for item in data.get("ranking", []):
            if str(item.get("passage_id")) == correct_passage_id:
                correct_rank = item["rank"]
                correct_passage_text = item["passage"]
                break
        ranks[qid] = {
            "rank": correct_rank,
            "passage_text": correct_passage_text
        }
    return ranks

def compare_baseline_vs_kg(baseline_ranks, kg_ranks, baseline_results):
    """Return only the cases where KG improves the rank."""
    improved_cases = []
    for qid, base_info in baseline_ranks.items():
        base_rank = base_info["rank"]
        kg_rank = kg_ranks.get(qid)  # kg_rank is an int

        if base_rank is not None and kg_rank is not None and kg_rank < base_rank:
            # collect full ranking text for baseline and KG
            baseline_text_ranking = [item["passage"] for item in baseline_results[qid]["ranking"]]
            # KG text ranking: only passages allowed by KG
            kg_text_ranking = [item["passage"] for item in baseline_results[qid]["ranking"] 
                               if str(item["passage_id"]) in allowed_ids_per_question.get(qid, [])]

            improved_cases.append({
                "question": base_info["question"],
                "baseline_rank": base_rank,
                "baseline_text_ranking": baseline_text_ranking,
                "kg_rank": kg_rank,
                "kg_text_ranking": kg_text_ranking
            })
    return improved_cases


def save_improved_cases_to_csv(city, model_name, improved_cases, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{city}_{model_name}_KG_improved_cases.csv")
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "question_text",
            "baseline_rank", "baseline_passage_id", "baseline_passage_text",
            "kg_rank", "kg_passage_id", "kg_passage_text"
        ])
        for case in improved_cases:
            writer.writerow([
                case['question_id'],
                case['question_text'],
                case['baseline_rank'],
                case['baseline_passage_id'],
                case['baseline_passage_text'],
                case['kg_rank'],
                case['kg_passage_id'],
                case['kg_passage_text']
            ])
    print(f"✅ KG improved cases saved to: {filename}")

def save_ranking_to_csv(city, model_name, rankings, explorer=None, output_dir="Results/RankingsCSV"):
    """
    Save rankings to CSV. If explorer is provided, also include filtered KG candidates.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{city}_{model_name}_ranking.csv")
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["QuestionID", "PassageID", "Rank", "Text", "FilteredByKG"])
        
        for qid, data in rankings.items():
            original_ranking = data.get("ranking", [])
            
            # Get allowed KG candidates if explorer is provided
            allowed_passages = set()
            if explorer:
                n_levels = 7
                top_k = 50
                candidates_by_level = explorer.extract_candidates(n_levels=n_levels, top_k=top_k, results=rankings)
                for level_cands in candidates_by_level.values():
                    allowed_passages.update(level_cands.get(qid, set()))
            
            for item in original_ranking:
                passage_id = item["passage_id"]
                text = item.get("text", "")
                rank = item["rank"]
                filtered_flag = 1 if passage_id in allowed_passages else 0
                writer.writerow([qid, passage_id, rank, text, filtered_flag])
    
    print(f"✅ Ranking saved to {csv_file}")

def save_comparison_to_file(city, baseline_path, kg_path, output_file, top_k=5):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_rankings = json.load(f)
    with open(kg_path, "r", encoding="utf-8") as f:
        kg_rankings = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for pid, data in baseline_rankings.items():
            question = data['question']
            baseline_ranking = data['ranking']
            kg_ranking = kg_rankings.get(pid, {}).get('ranking', [])

            correct_passage_id = pid
            baseline_rank = get_correct_answer_rank(baseline_ranking, correct_passage_id)
            kg_rank = get_correct_answer_rank(kg_ranking, correct_passage_id)

            f_out.write(f"\n=== Question: {question} ===\n")
            f_out.write(f"Correct passage ID: {correct_passage_id}\n")
            f_out.write(f"Baseline rank: {baseline_rank}\n")
            f_out.write(f"KG-based rank: {kg_rank}\n\n")

            f_out.write("Top passages in Baseline:\n")
            for r in baseline_ranking[:top_k]:
                f_out.write(f"  Rank {r['rank']}, ID: {r['passage_id']}, Score: {r['score']:.4f}\n")
                f_out.write(f"   {r['passage'][:200]}...\n")
            f_out.write("\nTop passages in KG-based method:\n")
            for r in kg_ranking[:top_k]:
                f_out.write(f"  Rank {r['rank']}, ID: {r['passage_id']}, Score: {r['score']:.4f}\n")
                f_out.write(f"   {r['passage'][:200]}...\n")
            f_out.write("\n" + "="*80 + "\n")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_dense(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])


def run_spade_experiment(city, model_name="sentence-transformers/all-mpnet-base-v2", output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning SPaDE experiment for: {city}")
    
    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dense_passage_embeddings = encode_dense(passages, tokenizer, model)
    dense_question_embeddings = encode_dense(questions, tokenizer, model)

    vectorizer = TfidfVectorizer().fit(passages + questions)
    sparse_passage_embeddings = vectorizer.transform(passages)
    sparse_question_embeddings = vectorizer.transform(questions)

    rankings = {}
    for i, pid in tqdm(enumerate(passage_ids), total=len(passage_ids), desc="SPaDE Ranking"):
        dense_scores = torch.nn.functional.cosine_similarity(
            dense_question_embeddings[i].unsqueeze(0), dense_passage_embeddings
        ).numpy()

        sparse_scores = cosine_similarity(
            sparse_question_embeddings[i], sparse_passage_embeddings
        ).flatten()

        combined_scores = 0.5 * dense_scores + 0.5 * sparse_scores
        top_indices = np.argsort(combined_scores)[::-1]

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(combined_scores[idx]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": data[pid]['question'], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"SPaDE-style rankings saved to '{output_path}'")


def run_bm25_experiment(city, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning BM25 experiment for: {city}")

    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)

    rankings = {}
    for i, pid in tqdm(enumerate(passage_ids), total=len(passage_ids), desc="BM25 Ranking"):
        query_tokens = questions[i].lower().split()
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(scores[idx]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": questions[i], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"BM25 rankings saved to '{output_path}'")

# Example: wrap the existing experiment function
def run_dense_cosine_experiment_with_csv(city, model_name, output_dir):
    results = run_dense_cosine_experiment(city, model_name=model_name, output_dir=output_dir)
    
    if results is None:
        raise ValueError("Experiment returned None. Make sure run_dense_cosine_experiment returns the ranking results.")
    
    save_ranking_to_csv(city, model_name, results, explorer=None, output_dir=os.path.join(output_dir, "RankingsCSV"))
    return results




def run_dense_cosine_experiment(city, model_name='all-mpnet-base-v2', output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning dense cosine experiment for: {city}")

    dataset_path = os.path.join("Dataset/qa/", "questionpassages_group_city.json")
    output_path = os.path.join(output_dir, f"{city}_ranking.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[city]

    passage_ids = list(data.keys())
    passages = [data[pid]['passage'] for pid in passage_ids]
    questions = [data[pid]['question'] for pid in passage_ids]

    model = SentenceTransformer(model_name)
    passage_embeddings = model.encode(passages, convert_to_tensor=True, show_progress_bar=True)
    query_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)

    rankings = {}
    cos_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)

    for i, pid in enumerate(passage_ids):
        scores = cos_scores[i]
        top_scores, top_indices = torch.topk(scores, k=len(passage_ids), largest=True)

        ranking = [{
            "rank": rank + 1,
            "passage_id": passage_ids[idx],
            "score": float(top_scores[rank]),
            "passage": passages[idx]
        } for rank, idx in enumerate(top_indices)]

        rankings[pid] = {"question": data[pid]['question'], "ranking": ranking}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"Dense cosine rankings saved to '{output_path}'")

    return rankings

import os
import json
import numpy as np
import csv
from collections import defaultdict
from Classes.embKG_explorer_classes import *

# ---------------------------
# IR Metrics
# ---------------------------
class InformationRetrievalMetrics:
    def compute_ir_metrics(self, rankings, k_values=[1, 5, 10, 20, 50, 100, 200]):
        num_samples = len(rankings)
        reciprocal_ranks = []
        hits_at_k = {k: 0 for k in k_values}

        for rank in rankings.values():
            reciprocal_ranks.append(1.0 / rank if rank else 0)
            for k in k_values:
                if rank and rank <= k:
                    hits_at_k[k] += 1

        mrr = np.mean(reciprocal_ranks)
        hits_at_k = {f"Hits@{k}": hits / num_samples for k, hits in hits_at_k.items()}
        metrics = {"MRR": mrr}
        metrics.update(hits_at_k)
        return metrics

# ---------------------------
# Utility Functions
# ---------------------------
def get_correct_passage_ranks(ranking_data):
    correct_passage_ranks = {}
    for qid, details in ranking_data.items():
        correct_passage_id = qid
        correct_rank = None
        for item in details.get("ranking", []):
            if str(item.get("passage_id")) == correct_passage_id:
                correct_rank = item["rank"]
                break
        correct_passage_ranks[qid] = correct_rank
    return correct_passage_ranks

def get_filtered_correct_passage_ranks_with_add(results_json, allowed_ids_per_question):
    correct_ranks = {}
    for question_id, data in results_json.items():
        correct_passage_id = question_id
        allowed_ids = set(map(str, allowed_ids_per_question.get(question_id, set())))
        
        original_ranking = data.get("ranking", [])
        allowed_items = [item for item in original_ranking if str(item["passage_id"]) in allowed_ids]
        disallowed_items = [item for item in original_ranking if str(item["passage_id"]) not in allowed_ids]
        
        reordered_items = allowed_items + disallowed_items
        # Reassign ranks based on the new order
        re_ranked = {str(item["passage_id"]): idx + 1 for idx, item in enumerate(reordered_items)}
        
        correct_ranks[question_id] = re_ranked.get(correct_passage_id, None)
    return correct_ranks

def get_filtered_correct_passage_ranks_without_add(results_json, allowed_ids_per_question):
    correct_ranks = {}
    for question_id, data in results_json.items():
        correct_passage_id = question_id
        allowed_ids = set(map(str, allowed_ids_per_question.get(question_id, set())))
        filtered_ranked_items = [
            item for item in data.get("ranking", []) if str(item["passage_id"]) in allowed_ids
        ]
        filtered_ranked_items.sort(key=lambda x: x["rank"])
        re_ranked = {str(item["passage_id"]): idx + 1 for idx, item in enumerate(filtered_ranked_items)}
        correct_ranks[question_id] = re_ranked.get(correct_passage_id, None)
    return correct_ranks

def report_not_found(correct_ranks, dict_ranks, arc_counts, n_levels):
    total_none_count = 0
    not_found_dict = defaultdict(list)
    not_found_by_level = defaultdict(list)

    for qid in correct_ranks:
        arc_count = arc_counts.get(qid, 0)
        levels_missing = []

        for level in range(n_levels):
            level_rank = dict_ranks.get(level, {}).get(qid)
            if level_rank is None:
                total_none_count += 1
                not_found_by_level[level].append(qid)
                levels_missing.append(level)

        if levels_missing:
            not_found_dict[qid] = {
                "levels": levels_missing,
                "arc_count": arc_count
            }

    return not_found_by_level

def format_metrics(metrics):
    return {key: f"{value:.5f}" for key, value in metrics.items()}

def save_metrics_to_csv(filename, data):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Top_K", "Level", "Metrics_All", "Metrics_Filtered", "Excluded_Zero_Arc"])
        for entry in data:
            writer.writerow(entry)

# ---------------------------
# Main Evaluation Logic
# ---------------------------
def evaluate_and_save(city, model_name, results, correct_ranks, explorer, arc_counts):
    ir_metrics = InformationRetrievalMetrics()
    output_data = []

    # ---- Baseline (All vs Filtered) ----
    metrics_all = ir_metrics.compute_ir_metrics(correct_ranks)
    metrics_all = format_metrics(metrics_all)

    filtered_ranks = {qid: rank for qid, rank in correct_ranks.items() if arc_counts.get(qid, 0) > 0}
    excluded = len(correct_ranks) - len(filtered_ranks)
    metrics_filtered = ir_metrics.compute_ir_metrics(filtered_ranks)
    metrics_filtered = format_metrics(metrics_filtered)
    output_data.append(["Nodes", "N/A", metrics_all, metrics_filtered, excluded])

    # ---- Multi-Level Evaluation ----
    

    top_k_values = [1, 3, 5, 10, 20, 50]      #numero di head da testare
    n_levels = 7                              #massimo numero di livelli

    for top_k in top_k_values:
        candidates_by_level = explorer.extract_candidates(
            n_levels=n_levels,
            top_k=top_k,
            results=results
        )
        for level, candidates in candidates_by_level.items():
            level_ranks = get_filtered_correct_passage_ranks_without_add(results, candidates)

            # Filter out zero-arc
            filtered_ranks_no_zero = {
                qid: rank for qid, rank in level_ranks.items()
                if arc_counts.get(qid, 0) > 0
            }
            excluded = len(level_ranks) - len(filtered_ranks_no_zero)

            metrics_all = format_metrics(ir_metrics.compute_ir_metrics(level_ranks))
            metrics_filtered = format_metrics(ir_metrics.compute_ir_metrics(filtered_ranks_no_zero))

            output_data.append([top_k, level, metrics_all, metrics_filtered, excluded])

    # ---- Save to CSV ----
    csv_filename = f"Results/Metrics/{city}_evaluation_results{model_name}.csv"
    save_metrics_to_csv(csv_filename, output_data)
    print(f"✅ Results saved to: {csv_filename}")

# ---------------------------
# Experiment Runner
# ---------------------------
import os
import csv
import json
from Classes.embKG_explorer_classes import *
from Classes.run_expermiment import *

# ---------------------------
# Ranking Saver
# ---------------------------
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

# ---------------------------
# Run City Experiment
# ---------------------------
def run_city_experiment(city, model_name, data_path=None, graph_path=None):
    print(f"\n========== Running for {city.upper()} ({model_name}) ==========")

    if data_path is None:
        data_path = os.path.join("Results/Rankings/"+model_name+"/", f"{city}_ranking.json")
    if graph_path is None:
        graph_path = os.path.join("", f"KG/{city.capitalize()}_Ps_KG.json")

    with open(data_path, "r") as f:
        results = json.load(f)

    # Initialize explorer for KG filtering
    explorer = Graph_Explorer(graph_path)
    
    # Save both baseline and KG-aware rankings
    save_ranking_to_csv(city, model_name, results, explorer=None, output_dir="Results/RankingsCSV/Baseline")
    save_ranking_to_csv(city, model_name, results, explorer=explorer, output_dir="Results/RankingsCSV/KG_Filtered")

    correct_ranks = get_correct_passage_ranks(results)
    arc_counts = explorer.get_arc_counts()

    evaluate_and_save(city, model_name, results, correct_ranks, explorer, arc_counts)


# ---------------------------
# Main
# ---------------------------
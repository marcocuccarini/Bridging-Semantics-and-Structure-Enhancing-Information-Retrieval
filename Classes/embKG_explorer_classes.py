import os
import json
import numpy as np
from collections import defaultdict

# ---------------------------
# Graph Explorer
# ---------------------------
class Graph_Explorer:
    def __init__(self, graph_file):
        graph_path =  graph_file
        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

    def extract_candidates(self, n_levels, top_k, results):
        results_level = {}
        seen_global = {}

        adjacency = defaultdict(set)
        for edge in self.graph:
            from_id = str(edge['from_id'])
            to_id = str(edge['to_id'])
            adjacency[from_id].add(to_id)
            adjacency[to_id].add(from_id)

        for level in range(n_levels):
            level_dict = {}
            for question_key, question_data in results.items():
                if question_key not in seen_global:
                    seen_global[question_key] = set()

                if level == 0:
                    top_k_candidates = question_data["ranking"][:top_k]
                    id_candidates = [str(item['passage_id']) for item in top_k_candidates]
                    seen_global[question_key].update(id_candidates)
                else:
                    previous_level_candidates = results_level[level - 1].get(question_key, [])
                    id_candidates = [str(item) for item in previous_level_candidates]

                new_candidates = set()
                for candidate_id in id_candidates:
                    neighbors = adjacency.get(candidate_id, set())
                    new_candidates.update(neigh for neigh in neighbors if neigh not in seen_global[question_key])

                seen_global[question_key].update(new_candidates)
                level_dict[question_key] = list(seen_global[question_key])

            results_level[level] = level_dict
        return results_level

    def get_arc_counts(self):
        arc_counts = defaultdict(int)
        for edge in self.graph:
            from_id = str(edge['from_id'])
            to_id = str(edge['to_id'])
            arc_counts[from_id] += 1
            arc_counts[to_id] += 1
        return arc_counts

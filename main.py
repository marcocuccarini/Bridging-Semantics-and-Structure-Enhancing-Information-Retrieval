from Classes.ranking_document import *
from Classes.run_expermiment import *

if __name__ == "__main__":
    output_folder = "Results/Rankings/"
    cities = ["florence", "rome", "venice"]
    dense_model_name = "all-mpnet-base-v2"   # only one dense model

    for city in cities:
        # 1️⃣ Run dense baseline
        output_dir = os.path.join(output_folder, "DenseBERT")
        run_dense_cosine_experiment(city, model_name=dense_model_name, output_dir=output_dir)

        # 2️⃣ Load results
        baseline_path = os.path.join(output_dir, f"{city}_ranking.json")
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

        baseline_ranks = get_correct_passage_ranks_with_text(baseline_results)

        # 3️⃣ KG-enhanced ranks
        graph_path = os.path.join("KG", f"{city.capitalize()}_Ps_KG.json")
        explorer = Graph_Explorer(graph_path)
        allowed_ids_per_question = explorer.extract_candidates(n_levels=7, top_k=5, results=baseline_results)
        kg_ranks = get_filtered_correct_passage_ranks_with_add(baseline_results, allowed_ids_per_question)

        # 4️⃣ Compare and save only improved cases
        improved_cases = compare_baseline_vs_kg(baseline_ranks, kg_ranks, baseline_results)
        save_improved_cases_to_csv(city, dense_model_name, improved_cases, output_dir)
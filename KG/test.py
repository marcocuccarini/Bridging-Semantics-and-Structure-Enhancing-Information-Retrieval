from sentence_transformers import SentenceTransformer, util
import json
import torch

# Load passages
with open("Florence_Ps_KG.json", "r") as f:
    connections = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to rank passages by similarity
def rank_passages(query, passages, top_k=10):
    if not passages:
        return []

    # Deduplicate by text
    unique_texts = {}
    for p in passages:
        txt = p['from_text'] + " ||| " + p['to_text']
        if txt not in unique_texts:
            unique_texts[txt] = p

    texts = list(unique_texts.keys())
    passages = list(unique_texts.values())

    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    cos_scores = util.cos_sim(q_emb, embeddings)[0]
    ranked_idx = torch.argsort(cos_scores, descending=True).tolist()

    # Select top-k
    top_ranked = []
    for i in ranked_idx[:top_k]:
        top_ranked.append({
            "from_id": passages[i]['from_id'],
            "to_id": passages[i]['to_id'],
            "text": texts[i],
            "similarity": cos_scores[i].item()
        })

    return top_ranked

# Function to filter top-k passages that can actually connect to a node
def filter_connectable(node_id, top_passages):
    filtered = []
    for p in top_passages:
        if p['from_id'] == node_id or p['to_id'] == node_id:
            filtered.append(p)
    return filtered

# Example queries
queries = [
    "History of Florence Cathedral",
    "Macchiaioli painters",
    "Brunelleschi architecture",
    "Medici family burial"
]

node_id = 0
top_k = 10  # number of top passages to consider

for q in queries:
    print(f"\n=== Query: {q} ===")
    top_passages = rank_passages(q, connections, top_k=top_k)  # Step 1: top-k by similarity
    connectable_passages = filter_connectable(node_id, top_passages)  # Step 2: filter by connection

    if not connectable_passages:
        print("⚠ No passages can connect to this node after top-k filtering.")
        continue

    for rank, p in enumerate(connectable_passages, 1):
        print(f"{rank}. From ID: {p['from_id']}, To ID: {p['to_id']}, Similarity: {p['similarity']:.4f}")
        print(f"   Text: {p['text'][:300]}...\n")

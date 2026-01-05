import os
try:
	import faiss  # type: ignore
except ImportError:
	raise ImportError(
		"The 'faiss' package is not installed. Install it with: pip install faiss-cpu"
	)

os.makedirs("faiss_index", exist_ok=True)

# The project uses sentence-transformers/all-MiniLM-L6-v2 which produces 384-d embeddings.
# If you use a different embedding model, change `d` accordingly in this file.
d = 384

index_path = os.path.join("faiss_index", "index.faiss")
if os.path.exists(index_path):
	print(f"FAISS index already exists at {index_path}; skipping creation.")
else:
	index = faiss.IndexFlatL2(d)
	faiss.write_index(index, index_path)
	print("Wrote faiss_index/index.faiss (dim=%d)" % d)
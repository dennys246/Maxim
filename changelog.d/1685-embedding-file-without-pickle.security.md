- **The semantic embedding file loads with pickle off** (#1685): `EmbeddingStore.load` used `allow_pickle=True`, so a
  crafted `util/semantic_embeddings.npz` ran code when loaded (it takes write access to your state directory). The
  file is now version 2, with string arrays only. An existing version 1 file is refused with one warning and the cache
  starts empty: memories captured before then drop out of semantic search, because nothing re-embeds them. Semantic
  similarity is off by default.

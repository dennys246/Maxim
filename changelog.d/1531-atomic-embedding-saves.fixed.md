- **Embedding saves are atomic, and an emptied store saves** (#1531): `EmbeddingStore.save` and `ComponentIndex.save`
  wrote their arrays straight to the target, so a kill mid-write tore the file; both go through
  `atomic_write_bytes` now. An empty `EmbeddingStore` used to skip the save, so embeddings removed during a session
  came back from the old file; it writes an empty file instead.

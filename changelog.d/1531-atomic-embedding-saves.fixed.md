- **Embedding saves are atomic, and an emptied store saves** (#1531): `EmbeddingStore.save` wrote straight to the
  target, so a kill mid-write tore the file; it goes through `atomic_write_bytes` now. An empty store used to skip the
  save, so embeddings removed during a session came back from the old file. It now writes an empty file after removing
  or clearing entries, or after refusing a version 1 file, and otherwise leaves the file alone, so a session that
  embedded nothing no longer has any effect on the machine-wide file another agent saved. In the public
  `ComponentIndex.save`, each file is written atomically too, and `load` rebuilds when the matrix and its sidecar
  disagree.

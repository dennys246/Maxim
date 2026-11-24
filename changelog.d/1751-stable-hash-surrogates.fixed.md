- `stable_hash_32` and `stable_hash_64_signed` now hash a string that contains a lone surrogate instead of raising.
  Every value they returned before is unchanged.

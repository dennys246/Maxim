- **`AgentConfig(with_bio_stack=True)` refuses `remembers=False` or `learns=False`** (#1225): the bio stack builds a
  Hippocampus and a NAc whatever those flags say, so `create_full_agent` used to undo them in silence. The
  contradiction now raises `maxim.exceptions.AgentConfigContradiction` (a `ValueError`) where the config is built.
  To migrate, drop `with_bio_stack` for a lightweight agent without one of them, or keep both flags `True`.

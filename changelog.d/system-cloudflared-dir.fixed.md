- **One reader for the system cloudflared config directory, so tests never see the host's tunnel**: role
  detection, the tunnel daemon spawner and `maxim doctor`'s tunnel-sync check each read `/etc/cloudflared/` on their
  own; all three now go through `utils/paths.py::cloudflared_system_dir`, which `MAXIM_CLOUDFLARED_SYSTEM_DIR` (an
  absolute path) can override. The test suite points both the override and the in-process default into its isolation
  root, and every child env a test builds keeps it (`tests/_child_env.py`), so a leader machine's real
  `/etc/cloudflared/config.yml` no longer turns role-dependent tests into leader runs. The doctor's sync check and the
  tunnel spawner's messages now name the path read.

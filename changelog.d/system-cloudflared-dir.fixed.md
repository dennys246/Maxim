- **One reader for the system cloudflared config directory, so tests never see the host's tunnel**: role
  detection, the tunnel daemon spawner and `maxim doctor`'s tunnel-sync check each read `/etc/cloudflared/` on their
  own; all three now go through `utils/paths.py::cloudflared_system_dir`, which `MAXIM_CLOUDFLARED_SYSTEM_DIR` can
  override. The test suite points it into its isolation root, so a leader machine's real `/etc/cloudflared/config.yml`
  no longer turns every role-dependent test into a leader run. The doctor's sync check now names the path it read.
  `maxim.tunnel.daemon_spawner.SYSTEM_CONFIG_PATH` is removed: it was fixed at import and ignored the override; call
  `resolve_config_path()` or `utils.paths.cloudflared_system_dir()` instead.

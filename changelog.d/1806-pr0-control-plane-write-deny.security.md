- **Agent file tools can no longer rewrite Maxim's own control-plane files** (#1806 PR 0, #1811): `write_file`,
  `edit_file` and the sandbox write tools now refuse, in every mode, any path under the resolved config directory
  (`~/.config/maxim/` or `$XDG_CONFIG_HOME/maxim/`: `mesh.yml`, `api_key`, `console_token`, `config.json`) and
  `robots.yaml`, `robots.yaml.bak` and `util/device_state.json` in the data home and the default `~/.maxim`.
  Before this, an agent started from your home directory could repoint or delete them. Code-execution tools
  (`bash`, `execute_file`, opted-in `run_tests`) are not covered; passive mode refuses those.
- **`create_sandbox_script` refuses a script name that is not a plain file name** (#1806 PR 0): a name with a path
  separator or a leading dot, or one that resolves outside the sandbox, is refused, and the script path gets the
  same `.git` and control-plane refusals as the other write tools.
- **`edit_file` now needs the same approval as `write_file`** for a write outside the workspace in supervised mode;
  the approval check only matched `write_file` before.

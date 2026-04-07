"""Unit tests for scan_cwd_tree() utility in filesystem_policy."""

from __future__ import annotations


from maxim.utils.filesystem_policy import scan_cwd_tree


class TestScanCwdTree:
    def test_basic_scan(self, tmp_path):
        """Scan returns files and directories."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# readme")

        entries = scan_cwd_tree(str(tmp_path), max_depth=2, max_entries=30)
        assert any("src/" in e for e in entries)
        assert any("main.py" in e for e in entries)
        assert any("README.md" in e for e in entries)

    def test_skips_noise_dirs(self, tmp_path):
        """.git, __pycache__, node_modules are skipped."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("gitconfig")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "express").mkdir()
        (tmp_path / "real_file.py").write_text("x = 1")

        entries = scan_cwd_tree(str(tmp_path), max_depth=2, max_entries=30)
        combined = "\n".join(entries)
        assert ".git" not in combined
        assert "__pycache__" not in combined
        assert "node_modules" not in combined
        assert "real_file.py" in combined

    def test_depth_limit(self, tmp_path):
        """Files beyond max_depth are not returned."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")
        (tmp_path / "top.txt").write_text("top")

        entries = scan_cwd_tree(str(tmp_path), max_depth=1, max_entries=30)
        combined = "\n".join(entries)
        assert "top.txt" in combined
        assert "deep.txt" not in combined

    def test_entry_limit(self, tmp_path):
        """No more than max_entries are returned."""
        for i in range(20):
            (tmp_path / f"file{i:02d}.txt").write_text(f"content {i}")

        entries = scan_cwd_tree(str(tmp_path), max_depth=1, max_entries=5)
        assert len(entries) == 5

    def test_dirs_before_files(self, tmp_path):
        """Directories appear before files in the output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "app.py").write_text("app")

        entries = scan_cwd_tree(str(tmp_path), max_depth=1, max_entries=30)
        dir_indices = [i for i, e in enumerate(entries) if e.rstrip().endswith("/")]
        file_indices = [i for i, e in enumerate(entries) if not e.rstrip().endswith("/")]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)

    def test_hidden_files_skipped(self, tmp_path):
        """Files starting with . are skipped."""
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.py").write_text("public")

        entries = scan_cwd_tree(str(tmp_path), max_depth=1, max_entries=30)
        combined = "\n".join(entries)
        assert ".hidden" not in combined
        assert "visible.py" in combined

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty directory returns empty list."""
        entries = scan_cwd_tree(str(tmp_path), max_depth=2, max_entries=30)
        assert entries == []

    def test_maxim_workspace_skipped(self, tmp_path):
        """.maxim_workspace is skipped in the CWD scan."""
        ws = tmp_path / ".maxim_workspace"
        ws.mkdir()
        (ws / "draft.py").write_text("draft")
        (tmp_path / "app.py").write_text("app")

        entries = scan_cwd_tree(str(tmp_path), max_depth=2, max_entries=30)
        combined = "\n".join(entries)
        assert ".maxim_workspace" not in combined
        assert "app.py" in combined

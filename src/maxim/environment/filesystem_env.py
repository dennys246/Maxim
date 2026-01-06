from __future__ import annotations

from .base import Environment
import os

class FileSystemEnv(Environment):
    def __init__(self, root):
        self.root = root
        self._done = False

    def reset(self):
        self._done = False
        return self.observe()

    def observe(self):
        return {
            "files": os.listdir(self.root),
            "cwd": self.root,
        }

    def step(self, result):
        if not result.success:
            return {"error": result.error}
        return self.observe()

    def is_done(self):
        return self._done

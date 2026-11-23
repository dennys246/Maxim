- **A lane pointed at a remote server is never silently dropped** (#1780): `lanes.medium.remote_url` or
  `lanes.small.remote_url` on a machine whose hardware has no such tier used to be ignored, with nothing logged.
  Only large had a placeholder. A remote URL now creates any of the three lanes.
- **`maxim peer connect --cloud` is accepted** (#1780): the parser used to reject it as an unknown option.

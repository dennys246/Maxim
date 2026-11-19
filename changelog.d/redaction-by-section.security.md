- **Cloud redaction renders private sections by identity**: a private prompt section's header is passed apart from
  its body, so the cloud redaction filter removes the whole section under the `strict` and `standard` policies, and
  a tool's result in the follow-up prompt is redacted according to that tool's sensitivity. Held inbound messages
  are now a private section too.

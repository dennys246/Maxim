- **Cloud redaction renders private sections by identity**: each private prompt section's header is passed apart from
  its body, and the body is shown in one neutral form on every backend (a line that reads like a section boundary
  is prefixed with `| `). Tool results in follow-up prompts carry a header that names the tool's sensitivity, and
  held inbound messages are acknowledged only once the model has been sent them.

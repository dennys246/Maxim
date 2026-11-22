- An unparseable follow-up input is no longer logged with its first 100 characters. That input can be a
  PRIVATE or SECRET tool's output, so the WARNING now carries only the input's length and a stable digest,
  and nothing from it reaches local log files (#1751).

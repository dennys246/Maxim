- **A sim's JSONL run log keeps what the run logs after shutdown** (#1476): to hide in-flight INFO lines from the
  terminal, the sim teardown raised the `maxim` logger to WARNING (#1588). That also cut the `MAXIM_LOG_FILE` JSONL, whose
  contract is DEBUG and up, for everything after shutdown, the final report and store saves included. It now raises
  every handler but the JSONL one (`utils/logging.py::quiet_console_handlers`: the console, a `log_file` and the
  leader's log buffer), so run logs also keep the INFO lines logged
  during shutdown.

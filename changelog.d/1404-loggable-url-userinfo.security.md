- **HTTP log events no longer carry a URL's username and password** (#1404): the URL that `http_request` and the
  default-on `http_request_failed` WARNING log kept `user:pass@` (only the query and fragment were stripped). Those
  events reach `MAXIM_LOG_FILE` and the leader proxy's remotely readable log buffer. The host is now logged without
  its userinfo, and a URL that cannot be parsed is logged as `<url>` instead of raising inside the failure path.

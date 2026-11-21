- **A sim's `actions.jsonl` no longer records a URL's credentials, path or query string** (#1404): tool arguments,
  results, errors and block reasons were written verbatim, so a URL the model fetched (with a token in its query) or
  an httpx error carrying a request URL with `user:pass@` could land in a record committed with experiment evidence.
  Every `scheme://` URL in those fields (keys included) now keeps only its `scheme://host[:port]`, as `report.json`
  already does (#997).
  Text without a URL is unchanged. The console's run view shows the reduced URLs.

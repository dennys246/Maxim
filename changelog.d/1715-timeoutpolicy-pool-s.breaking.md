- **`maxim.utils.http.TimeoutPolicy.total_s` is renamed `pool_s`** (#1715). It only ever bounded the wait for a
  pooled connection (httpx's `pool`), but nearly every caller set it as if it bounded the whole request. No
  behaviour changes. Code that constructs `TimeoutPolicy(total_s=...)` gets a `TypeError` and should pass
  `pool_s=` instead.

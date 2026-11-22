- Reachy Mini: `ReachyMiniController.disconnect()` now closes the SDK's WebSocket client, as the SDK's own `__exit__`
  does, instead of leaving its thread to garbage collection, and a connect that fails drops its handle, so the next
  disconnect no longer retries a re-acquire on a dead one. A camera-backend connect onto a daemon whose media is
  already released (a killed `no_media` client) now warns and names `POST /api/media/acquire` instead of failing
  later with a bare `Errno 61`. The SDK-only orient scripts (`LiveRig` and its reconnect, `live_1_smoke`) re-acquire
  the media they released on every exit but SIGKILL (SIGTERM and SIGHUP now exit normally there), and the motion lint refuses a `no_media` handle that its file
  never undoes (#1712).

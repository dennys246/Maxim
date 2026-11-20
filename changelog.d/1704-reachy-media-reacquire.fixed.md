- **`ReachyMiniController` no longer leaves the robot's camera released after a `no_media` connection** (#1704):
  reachy-mini 1.8.3's `no_media` connect releases the daemon's camera and microphone for the whole robot, and only
  the SDK's own context exit took them back. `disconnect()` now re-acquires the media its own connect released,
  before an SSH tunnel stops, and a connect that fails after the release undoes it too. Before, every later camera
  client, the Maxim app included, failed with "Connection refused" until a manual `POST /api/media/acquire`.
  - A release another client made is left alone.
  - When the state before the connect is unknown, or a re-acquire can't be confirmed, the controller logs a WARNING
    naming that endpoint.
  - An unconfirmed re-acquire is retried at the next disconnect after a reconnect, provided that reconnect can read
    the daemon's status. If it can't, the controller drops the claim and logs another WARNING rather than guess.
  - Scripts that build their own SDK handle with `no_media`, outside the controller, still leak the release (#1712).

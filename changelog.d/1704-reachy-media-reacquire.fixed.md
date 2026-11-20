- **A `no_media` Reachy connection no longer leaves the robot's camera released** (#1704): reachy-mini 1.8.3's
  `no_media` connect releases the daemon's camera and microphone for the whole robot, and only the SDK's own context
  exit took them back. `ReachyMiniController.disconnect()` now re-acquires the media its own connect released.
  Before, every later camera client, the Maxim app included, failed with "Connection refused" until a manual
  `POST /api/media/acquire`. A release another client made is left alone. When the state before the connect is
  unknown, or the re-acquire fails, the controller logs a WARNING naming that endpoint.

- **A rotated API key for a remote lane takes effect without a restart** (#1371): a lane's key also travelled
  through an internal environment variable (`MAXIM_LANE_<TIER>_API_KEY`, and `MAXIM_PLACEMENT_<KEY>_API_KEY` for a
  fallback placement) written with `setdefault`, so a value already there (an earlier build in the same process, or
  one inherited from a parent) beat a key rotated in config since. The resolved key now rides on the lane's provider
  entry, which the backends send. To override a lane's key from the environment, set
  `MAXIM_LANE_<TIER>_REMOTE_API_KEY`, as `maxim tunnel key export` prints; the old internal variable is ignored, with
  a one-time warning when it differs.

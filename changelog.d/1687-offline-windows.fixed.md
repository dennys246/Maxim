- **Maxim imports on Windows again** (#1687): `maxim._offline` used `socket.AF_UNIX`, which Windows CPython lacks, so
  importing it raised, and with it `maxim.tools.base` and everything else that loads the offline latch. The latch's
  allowed socket families now include `AF_UNIX` only where the platform has it. The latch stays POSIX-only:
  `engage()` on any other platform raises `OfflineUnsupportedPlatform` before the latch goes up, so a Windows process
  started with `MAXIM_OFFLINE` set fails at `import maxim`, and `maxim.advisor.open()` refuses with
  `AdvisorCompositionError`, instead of reporting offline on a platform whose refusal set has never been checked.

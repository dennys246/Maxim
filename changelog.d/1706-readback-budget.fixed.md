- **A Reachy move no longer refuses a healthy robot's pose read for taking more than a third of the deadline**
  (#1706): the fresh readback split `FRESH_READ_DEADLINE_S` (100 ms) into three equal httpx timeouts, but the third
  one (`total_s`) only bounds the wait for a pooled connection. So the connect and the read each got about 33 ms,
  and reads of 35–43 ms from a Mac refused moves with `reference`. Each phase may now use the whole deadline, and
  the check after the read still refuses any reference older than it. The constant is unchanged until it is
  re-measured on the robot, but the read budget actually enforced goes from about 33 ms to the full 100 ms (300 ms
  tunnelled).
  - A read that fails can now block for about 3–4 times the deadline before it is refused (twice that with the
    retry between chunks): about 0.3–0.4 s on the LAN, against about 0.1 s before.
  - `TimeoutPolicy`'s docstring and `docs/troubleshooting/http_debugging.md` now say what `total_s` does: it bounds
    the pool wait, not the whole request, and DNS is not timed.

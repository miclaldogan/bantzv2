## 2025-02-28 - asyncio.gather for Network Boundaries
**Learning:** In async environments wrapping synchronous APIs via `loop.run_in_executor` (like the Gmail tool wrapping standard google-api-python-client calls), developers often default to sequential `await`ing. This creates an N+1 latency bottleneck where total duration is the sum of all calls.
**Action:** Always scan for grouped network calls across thread executors and aggregate them using `asyncio.gather(..., return_exceptions=True)`. Ensure exceptions are properly evaluated post-gather to maintain original retry/fallback logic.

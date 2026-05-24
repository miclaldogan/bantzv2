Why did `a._last_analysis = 0` cause `feed_frames` to return `None` in `test_interval_gating`?
In `feed_frames`:
```python
        now = time.monotonic()

        # Should we start accumulating?
        if not self._accumulating:
            if now - self._last_analysis < self._sample_interval_s:
                return None  # not time yet
```
If `now` is `1651.6` and `a._last_analysis = 0`, then `now - 0` is `1651.6`.
If `self._sample_interval_s` is `60.0`, then `1651.6 < 60.0` is False, so it SHOULD start accumulating.
Wait, maybe `time.monotonic()` returned something less than `60.0`?
When the tests ran on GitHub CI, the VM might have just booted! So `time.monotonic()` could literally be `< 60` seconds!
Yes! `time.monotonic()` returns the time since boot. If the runner VM was started recently (e.g. 10-20 seconds ago), `time.monotonic()` is `~20.0`.
If `a._last_analysis = 0`, `now - 0` is `20.0`.
`20.0 < 60.0` is True, so `return None`!

That's an amazing insight! The test assumed `time.monotonic() > 60` or mocked time. It didn't mock time, and used `a._last_analysis = 0`.
To fix this, we should set `a._last_analysis = time.monotonic() - 100.0`.

Now, how about `test_returns_none_for_simple_request`?
```python
            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))
```
It crashed because `data_layer` was not mocked, so `data_layer.conversations` is None.
Wait, `bantz/core/routing_engine.py` does:
```python
from bantz.data import data_layer
...
data_layer.conversations.add(...)
```
So we should mock `data_layer` in `test_returns_none_for_simple_request`.

And `test_execute_plan_without_history_still_works`:
It mocks `data_layer`:
```python
        with patch("bantz.core.routing_engine.registry") as mock_reg, \
             patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.core.routing_engine.data_layer") as dl:
```
But it expects `result is None`, whereas `execute_plan` now returns `BrainResult(...)`.
Why does `execute_plan` return `BrainResult` instead of `None`?
In `bantz/core/routing_engine.py`:
```python
    if not steps:
        msg = "I couldn't break that into actionable steps. Try being more specific."
        log.warning("execute_plan: planner returned no steps for: %.80s", en_input)
        data_layer.conversations.add("assistant", msg, tool_used="planner")
        return BrainResult(response=msg, tool_used="planner")
```
It explicitly returns `BrainResult`. So the test assertion `assert result is None` is just out of date!

Let's fix these three tests!

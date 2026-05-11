1.  **Analyze `live_ui.py`'s `_probe_services`:**
    *   The `_probe_services` method in `src/bantz/interface/live_ui.py` is an asynchronous function that probes various services sequentially.
    *   This is a known performance issue: "For performance in async components doing multiple independent network requests or health checks (e.g., `_probe_services` in `live_ui.py`), execute them concurrently using `asyncio.gather` alongside a shared `httpx.AsyncClient` rather than awaiting them sequentially."

2.  **Optimize `_probe_services`:**
    *   Modify `_probe_services` to create a single `httpx.AsyncClient`.
    *   Refactor the individual service checks (Ollama, Neo4j, Palace, SQLite, GPS, Desktop, Core) into separate async helper functions.
    *   Use `asyncio.gather` to run these checks concurrently.
    *   Update `self._services` and log messages based on the results.

3.  **Ensure Pre-commit Steps:**
    *   Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

4.  **Submit PR:**
    *   Submit a PR with the title "⚡ Bolt: [performance improvement]" and a detailed description following the Bolt persona guidelines.

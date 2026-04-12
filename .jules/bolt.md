## 2024-05-18 - [HTTP Connection Pooling for LLM Clients]
**Learning:** Initial implementation of the LLM clients (`OllamaClient` in `src/bantz/llm/ollama.py` and `GeminiClient` in `src/bantz/llm/gemini.py`) were re-instantiating an `httpx.AsyncClient` for *every* individual request. This bypasses HTTP connection pooling, establishing a new TCP connection on every prompt, which introduces measurable and completely unnecessary latency in a local AI assistant that fires many rapid, sequential LLM calls.
**Action:** Always verify if HTTP clients (like `httpx` or `requests`) are being reused. Use a shared, lazy-loaded client instance (e.g. `self.client`) to leverage connection pooling. Pass varying configuration needs like `timeout` directly to the request method (e.g. `self.client.post(..., timeout=30.0)`) instead of hardcoding it during the client's instantiation.

## 2024-05-18 - [File System Traversal Optimization]
**Learning:** In high-frequency polling scenarios (like scanning the `/proc` filesystem for process IDs), using `pathlib.Path.iterdir()` introduces significant overhead compared to `os.listdir()` due to the instantiation of a `Path` object for every directory entry. When scanning thousands of items, this object creation cost adds up unnecessarily.
**Action:** Use `os.listdir()` and raw strings for paths when iterating over large directories in performance-critical or frequently-polled code paths, replacing `Path` object instantiation where appropriate.

## 2026-04-12 - [N+1 Query Optimization in SQLite Migrations]
**Learning:** The data migration script for converting JSON files into the new SQLite format (`src/bantz/data/migration.py`) was previously executing thousands of individual `conn.execute()` insert statements within `for` loops. This N+1 query pattern creates significant overhead during the initial database population, blocking the application startup.
**Action:** Always identify loops that execute individual database inserts or updates. Refactor these to accumulate the data into lists and execute a single `conn.executemany()` to drastically reduce the SQLite statement compilation and execution overhead.

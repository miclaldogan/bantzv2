## 2024-05-18 - [HTTP Connection Pooling for LLM Clients]
**Learning:** Initial implementation of the LLM clients (`OllamaClient` in `src/bantz/llm/ollama.py` and `GeminiClient` in `src/bantz/llm/gemini.py`) were re-instantiating an `httpx.AsyncClient` for *every* individual request. This bypasses HTTP connection pooling, establishing a new TCP connection on every prompt, which introduces measurable and completely unnecessary latency in a local AI assistant that fires many rapid, sequential LLM calls.
**Action:** Always verify if HTTP clients (like `httpx` or `requests`) are being reused. Use a shared, lazy-loaded client instance (e.g. `self.client`) to leverage connection pooling. Pass varying configuration needs like `timeout` directly to the request method (e.g. `self.client.post(..., timeout=30.0)`) instead of hardcoding it during the client's instantiation.

## 2024-05-18 - [File System Traversal Optimization]
**Learning:** In high-frequency polling scenarios (like scanning the `/proc` filesystem for process IDs), using `pathlib.Path.iterdir()` introduces significant overhead compared to `os.listdir()` due to the instantiation of a `Path` object for every directory entry. When scanning thousands of items, this object creation cost adds up unnecessarily.
**Action:** Use `os.listdir()` and raw strings for paths when iterating over large directories in performance-critical or frequently-polled code paths, replacing `Path` object instantiation where appropriate.

## 2024-05-18 - [SQLite Batch Inserts (N+1 Optimization)]
**Learning:** In `src/bantz/data/sqlite_store.py`, multiple store classes (`SQLiteProfileStore`, `SQLitePlaceStore`, `SQLiteScheduleStore`, `SQLiteSessionStore`) were found using a loop of `conn.execute(...)` statements to insert data during bulk save operations (e.g., table rebuilds). This causes an N+1 query performance bottleneck during serialization.
**Action:** Always use `conn.executemany(...)` along with a list comprehension for bulk insert operations in SQLite to take advantage of SQLite's C-level loop, providing significant performance gains during bulk data saving operations.

## 2024-05-18 - [Optimized process scanning with os.scandir]
**Learning:** For high-performance I/O loops iterating over large directories like `/proc` in Python, `os.scandir()` is measurably faster than `os.listdir()` and `pathlib.Path.iterdir()`. `os.scandir()` yields cached `DirEntry` objects instead of allocating a full list of strings upfront, reducing object instantiation overhead.
**Action:** Use `with os.scandir(path) as it: for entry in it:` when scanning thousands of files/directories, especially if only checking properties like `entry.name.isdigit()` or `entry.is_dir()`.

## 2024-05-18 - [SQLite Bulk Insertion Optimization]
**Learning:** In `src/bantz/data/migration.py`, data migration functions were looping over dictionaries and running `conn.execute(...)` for each key-value pair or list element. This creates a classic N+1 query performance bottleneck when migrating large datasets.
**Action:** Replace `conn.execute(...)` loops with a list comprehension paired with `conn.executemany(...)` to run the operations in bulk, reducing database round-trips and significantly improving data loading performance.

## 2024-05-18 - [SQLite Table Existence Check Optimization]
**Learning:** To check if a table is empty or has data in SQLite, `SELECT COUNT(*) FROM table` performs an O(N) full table/index scan. This becomes a performance bottleneck as the table grows.
**Action:** Use `SELECT 1 FROM table LIMIT 1` combined with `fetchone() is not None` instead. This is an O(1) operation that returns immediately after finding the first row, avoiding full scans.
## 2025-02-12 - Concurrent HTTP Fetching Optimization
**Learning:** Sequential HTTP calls (N+1) using `await` inside a loop are a major bottleneck for network-bound tools in Bantz.
**Action:** When performing multiple independent HTTP requests (like fetching API items), group them using `asyncio.gather` with `return_exceptions=True` instead of sequential `await` calls.

## $(date +%Y-%m-%d) - Eliminate N+1 Network Latency in Telegram Broadcasts
**Learning:** Sequential `await` calls in loops for independent HTTP requests (like sending notifications to multiple Telegram users) cause N+1 network latency, blocking the event loop and scaling poorly with the number of requests.
**Action:** Always group independent I/O-bound async tasks using a list comprehension and execute them concurrently via `asyncio.gather(*coros, return_exceptions=True)`. This reduces execution time from O(N) to O(1) in terms of latency, while preserving the isolation of individual request failures.

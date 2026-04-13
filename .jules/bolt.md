## 2024-05-18 - [Optimized process scanning with os.scandir]
**Learning:** For high-performance I/O loops iterating over large directories like `/proc` in Python, `os.scandir()` is measurably faster than `os.listdir()` and `pathlib.Path.iterdir()`. `os.scandir()` yields cached `DirEntry` objects instead of allocating a full list of strings upfront, reducing object instantiation overhead.
**Action:** Use `with os.scandir(path) as it: for entry in it:` when scanning thousands of files/directories, especially if only checking properties like `entry.name.isdigit()` or `entry.is_dir()`.

## 2024-05-18 - [SQLite Bulk Insertion Optimization]
**Learning:** In `src/bantz/data/migration.py`, data migration functions were looping over dictionaries and running `conn.execute(...)` for each key-value pair or list element. This creates a classic N+1 query performance bottleneck when migrating large datasets.
**Action:** Replace `conn.execute(...)` loops with a list comprehension paired with `conn.executemany(...)` to run the operations in bulk, reducing database round-trips and significantly improving data loading performance.

## 2024-05-18 - [SQLite `exists()` Optimization]
**Learning:** In `src/bantz/data/sqlite_store.py`, existence checks for tables were implemented using `SELECT COUNT(*) FROM table`. For large tables, `COUNT(*)` performs a full index or table scan to compute the exact number of rows, causing O(N) performance degradation as data grows.
**Action:** Replaced `SELECT COUNT(*)` with `SELECT 1 FROM table LIMIT 1`. This stops query execution as soon as the first row is found, dropping the algorithmic complexity from O(N) to O(1) and resulting in a measurably faster existence check (~6x speedup in local testing).

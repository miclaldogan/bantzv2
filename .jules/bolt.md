## 2024-05-18 - [Optimized process scanning with os.scandir]
**Learning:** For high-performance I/O loops iterating over large directories like `/proc` in Python, `os.scandir()` is measurably faster than `os.listdir()` and `pathlib.Path.iterdir()`. `os.scandir()` yields cached `DirEntry` objects instead of allocating a full list of strings upfront, reducing object instantiation overhead.
**Action:** Use `with os.scandir(path) as it: for entry in it:` when scanning thousands of files/directories, especially if only checking properties like `entry.name.isdigit()` or `entry.is_dir()`.

## 2024-05-18 - [SQLite Bulk Insertion Optimization]
**Learning:** In `src/bantz/data/migration.py`, data migration functions were looping over dictionaries and running `conn.execute(...)` for each key-value pair or list element. This creates a classic N+1 query performance bottleneck when migrating large datasets.
**Action:** Replace `conn.execute(...)` loops with a list comprehension paired with `conn.executemany(...)` to run the operations in bulk, reducing database round-trips and significantly improving data loading performance.

## 2024-05-18 - [SQLite Table Existence Check Optimization]
**Learning:** To check if a table is empty or has data in SQLite, `SELECT COUNT(*) FROM table` performs an O(N) full table/index scan. This becomes a performance bottleneck as the table grows.
**Action:** Use `SELECT 1 FROM table LIMIT 1` combined with `fetchone() is not None` instead. This is an O(1) operation that returns immediately after finding the first row, avoiding full scans.

## 2024-05-18 - Avoid N+1 Updates in Loop Check Functions
**Learning:** Database interactions inside a loop such as `conn.execute("UPDATE ...")` within `for row in rows:` in frequently called checking functions (e.g., `check_due` and `check_place_due` in `scheduler.py`) can create massive performance bottlenecks as the number of checks increases (N+1 query problem). This is particularly impactful for recurring tasks and location-based checks which might scan multiple reminders.
**Action:** Accumulate item IDs or parameter tuples within the loop, then use a single `conn.executemany` statement outside the loop to process all updates in bulk. Always look for `execute` inside loops in data-handling code as a prime target for bulk operation optimization.

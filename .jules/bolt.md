## 2024-05-18 - [Optimized process scanning with os.scandir]
**Learning:** For high-performance I/O loops iterating over large directories like `/proc` in Python, `os.scandir()` is measurably faster than `os.listdir()` and `pathlib.Path.iterdir()`. `os.scandir()` yields cached `DirEntry` objects instead of allocating a full list of strings upfront, reducing object instantiation overhead.
**Action:** Use `with os.scandir(path) as it: for entry in it:` when scanning thousands of files/directories, especially if only checking properties like `entry.name.isdigit()` or `entry.is_dir()`.

## 2024-05-18 - [SQLite Bulk Insertion Optimization]
**Learning:** In `src/bantz/data/migration.py`, data migration functions were looping over dictionaries and running `conn.execute(...)` for each key-value pair or list element. This creates a classic N+1 query performance bottleneck when migrating large datasets.
**Action:** Replace `conn.execute(...)` loops with a list comprehension paired with `conn.executemany(...)` to run the operations in bulk, reducing database round-trips and significantly improving data loading performance.

## 2024-05-18 - [SQLite O(1) Existence Checks]
**Learning:** Using `SELECT COUNT(*) FROM table` to check if any rows exist in SQLite performs a full table or index scan, which takes O(N) time and is unnecessarily slow, particularly as datasets grow.
**Action:** Use `SELECT 1 FROM table LIMIT 1` for existence checks to perform the operation in O(1) time, as it returns immediately upon finding the first row without scanning the rest of the table.

git checkout -b bolt-concurrent-hn-fetches
git add src/bantz/tools/news.py .jules/bolt.md
git commit -m "⚡ Bolt: Implement concurrent Hacker News API fetches"

gh pr create --title "⚡ Bolt: Implement concurrent Hacker News API fetches" --body "💡 What: Modified \`_fetch_hn\` in \`src/bantz/tools/news.py\` to fetch Hacker News story items concurrently using \`asyncio.gather\` with \`return_exceptions=True\`.
🎯 Why: The previous implementation iterated over story IDs and called \`await client.get(...)\` sequentially, creating an N+1 API fetching bottleneck.
📊 Impact: This significantly reduces the total fetch time for the news tool, making it substantially faster without altering application architecture.
🔬 Measurement: Verify by executing \`await tool.execute(source=\"hn\")\` or running benchmarks for concurrent fetching compared to sequential."

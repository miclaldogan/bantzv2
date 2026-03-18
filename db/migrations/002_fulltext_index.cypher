// Bantz — Migration 002: full-text search index
// Enables fast keyword queries across all textual node properties.
//
// The index name "bantz_fulltext" is referenced in MemoryManager.query().

CALL db.index.fulltext.createNodeIndex(
  "bantz_fulltext",
  ["Person", "Topic", "Decision", "Task", "Event",
   "Location", "Document", "Reminder", "Commitment", "Project", "Fact"],
  ["name", "title", "description", "what", "text", "path", "context"]
);

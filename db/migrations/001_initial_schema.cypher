// Bantz — Migration 001: initial schema
// Creates indexes for all node labels and the full-text search index.

// ── Node indexes ──────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS FOR (p:Person)     ON (p.name);
CREATE INDEX IF NOT EXISTS FOR (t:Topic)      ON (t.name);
CREATE INDEX IF NOT EXISTS FOR (d:Decision)   ON (d.what);
CREATE INDEX IF NOT EXISTS FOR (tk:Task)      ON (tk.description);
CREATE INDEX IF NOT EXISTS FOR (e:Event)      ON (e.title);
CREATE INDEX IF NOT EXISTS FOR (l:Location)   ON (l.name);
CREATE INDEX IF NOT EXISTS FOR (dc:Document)  ON (dc.path);
CREATE INDEX IF NOT EXISTS FOR (r:Reminder)   ON (r.title);
CREATE INDEX IF NOT EXISTS FOR (c:Commitment) ON (c.what);
CREATE INDEX IF NOT EXISTS FOR (pr:Project)   ON (pr.name);
CREATE INDEX IF NOT EXISTS FOR (f:Fact)       ON (f.text);

-- Entity Memory Database Schema

CREATE TABLE IF NOT EXISTS thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    thought_type TEXT CHECK(thought_type IN ('thesis', 'antithesis', 'synthesis', 'reflection', 'decision')),
    content TEXT NOT NULL,
    triad_id INTEGER,
    parent_thought_id INTEGER,
    model_used TEXT,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    action_type TEXT NOT NULL,
    target TEXT,
    description TEXT,
    success BOOLEAN DEFAULT 0,
    error_message TEXT,
    thought_id INTEGER
);

CREATE TABLE IF NOT EXISTS budget (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    opus_granted_usd REAL DEFAULT 5.0,
    opus_spent_usd REAL DEFAULT 0.0,
    gemini_granted_usd REAL DEFAULT 3.0,
    gemini_spent_usd REAL DEFAULT 0.0,
    total_api_calls INTEGER DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO budget (id) VALUES (1);

CREATE TABLE IF NOT EXISTS identity (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    chosen_name TEXT,
    description TEXT,
    nostr_kind0_published BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    evolved_at DATETIME
);

CREATE TABLE IF NOT EXISTS capabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'not_started',
    file_path TEXT,
    first_attempt DATETIME,
    last_success DATETIME,
    times_used INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS nostr_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    direction TEXT CHECK(direction IN ('sent', 'received')),
    event_id TEXT,
    kind INTEGER,
    pubkey_hex TEXT,
    content TEXT,
    relay TEXT,
    success BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cycle_number INTEGER,
    duration_seconds REAL,
    thoughts_generated INTEGER DEFAULT 0,
    actions_taken INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    error TEXT,
    mode TEXT DEFAULT 'triad'
);

CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    category TEXT NOT NULL,
    insight TEXT NOT NULL,
    source_cycle INTEGER,
    confidence REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS entity_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL,
    status TEXT CHECK(status IN ('active', 'completed', 'failed', 'abandoned')) DEFAULT 'active',
    source_cycle INTEGER,
    completed_cycle INTEGER
);

CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT DEFAULT 'seed'
);

CREATE INDEX IF NOT EXISTS idx_thoughts_timestamp ON thoughts(timestamp);
CREATE INDEX IF NOT EXISTS idx_thoughts_triad ON thoughts(triad_id);
CREATE INDEX IF NOT EXISTS idx_learnings_cat ON learnings(category);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);

CREATE VIEW IF NOT EXISTS budget_status AS
SELECT
    opus_granted_usd,
    opus_spent_usd,
    ROUND(opus_granted_usd - opus_spent_usd, 4) as opus_remaining,
    gemini_granted_usd,
    gemini_spent_usd,
    ROUND(gemini_granted_usd - gemini_spent_usd, 4) as gemini_remaining,
    total_api_calls,
    last_updated
FROM budget;

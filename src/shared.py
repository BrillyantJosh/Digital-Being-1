#!/usr/bin/env python3
"""
SHARED — Skupne funkcije za digitalno bitje.
Uporablja heartbeat.py, communication_cycle.py in angel_observer.py.

Vsebuje: konstante, config, DB, Gemini API, persistence, code execution, task management.
"""

import os
import re
import json
import time
import sqlite3
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# KONSTANTE IN CONFIG
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/opt/entity")
STATE_DIR = BASE_DIR / "state"
SECRETS_DIR = BASE_DIR / "secrets"
LOGS_DIR = BASE_DIR / "logs"
THOUGHTS_DIR = LOGS_DIR / "thoughts"
CAPABILITIES_DIR = BASE_DIR / "capabilities"
SRC_DIR = BASE_DIR / "src"

DB_PATH = STATE_DIR / "memory.db"
DMS_DB_PATH = STATE_DIR / "dms.db"
GENESIS_PATH = BASE_DIR / "GENESIS.md"
LOCK_FILE = STATE_DIR / "heartbeat.lock"

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_PRICE_INPUT = 0.10 / 1_000_000
GEMINI_PRICE_OUTPUT = 0.40 / 1_000_000

ACTION_TIMEOUT = 30  # sekund
PROTECTED_FILES = [
    str(GENESIS_PATH),
    str(DB_PATH),
    str(SRC_DIR / "heartbeat.py"),
    str(SRC_DIR / "shared.py"),
    str(SRC_DIR / "communication_cycle.py"),
]

# ── Config loader — en vir resnice za ustvarjalca ──
def _load_config():
    config_path = BASE_DIR / "config" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

_CONFIG = _load_config()
CREATOR_NAME = _CONFIG.get("creator_name", "Creator")
CREATOR_PUBKEY_HEX = _CONFIG.get("nostr", {}).get("creator_pubkey_hex", "")
CREATOR_REPLY_FILENAME = f"{CREATOR_NAME.upper()}_REPLY.md"


# ══════════════════════════════════════════════════════════════════════════
# LOGIRANJE
# ══════════════════════════════════════════════════════════════════════════

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


# ══════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # za concurrent access
    conn.execute("PRAGMA busy_timeout=5000")  # čakaj do 5s če je locked
    return conn


def init_database():
    if not DB_PATH.exists():
        schema_path = STATE_DIR / "schema.sql"
        if schema_path.exists():
            conn = sqlite3.connect(DB_PATH)
            conn.executescript(schema_path.read_text())
            conn.commit()
            conn.close()
            log("Database initialized")


def migrate_db():
    """Dodaj nove stolpce/tabele."""
    conn = get_db()

    # heartbeats.mode stolpec
    try:
        conn.execute("ALTER TABLE heartbeats ADD COLUMN mode TEXT DEFAULT 'triad'")
        log("MIGRACIJA: Dodan stolpec heartbeats.mode")
    except:
        pass

    # learnings tabela
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT NOT NULL,
            insight TEXT NOT NULL,
            source_cycle INTEGER,
            confidence REAL DEFAULT 1.0
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_learnings_cat ON learnings(category)")
    except:
        pass

    # entity_state tabela
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS entity_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    except:
        pass

    # migracija: josh_reply_hash → creator_reply_hash
    try:
        conn.execute("UPDATE entity_state SET key='creator_reply_hash' WHERE key='josh_reply_hash'")
    except:
        pass

    # goals tabela
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT NOT NULL,
            status TEXT CHECK(status IN ('active', 'completed', 'failed', 'abandoned')) DEFAULT 'active',
            source_cycle INTEGER,
            completed_cycle INTEGER
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
    except:
        pass

    # knowledge tabela
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT DEFAULT 'seed'
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic)")
    except:
        pass

    # conversation_log tabela
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT NOT NULL CHECK(direction IN ('incoming', 'outgoing', 'outgoing_pending')),
            content TEXT NOT NULL,
            nostr_event_id TEXT UNIQUE,
            timestamp TEXT DEFAULT (datetime('now')),
            responded_to INTEGER DEFAULT 0,
            contact_type TEXT DEFAULT 'creator',
            tasks_extracted INTEGER DEFAULT 0
        )""")
    except:
        pass

    # conversation_log — dodaj stolpca če manjkata (obstoječi DB)
    try:
        conn.execute("ALTER TABLE conversation_log ADD COLUMN contact_type TEXT DEFAULT 'creator'")
    except:
        pass
    try:
        conn.execute("ALTER TABLE conversation_log ADD COLUMN tasks_extracted INTEGER DEFAULT 0")
    except:
        pass

    # conversation_log — posodobi CHECK constraint za 'outgoing_pending'
    try:
        # Preveri ali stara tabela nima outgoing_pending v CHECK constraintu
        table_info = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='conversation_log'").fetchone()
        if table_info and 'outgoing_pending' not in (table_info[0] or ''):
            log("MIGRACIJA: Posodabljam conversation_log CHECK constraint...")
            conn.execute("""CREATE TABLE IF NOT EXISTS conversation_log_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL CHECK(direction IN ('incoming', 'outgoing', 'outgoing_pending')),
                content TEXT NOT NULL,
                nostr_event_id TEXT UNIQUE,
                timestamp TEXT DEFAULT (datetime('now')),
                responded_to INTEGER DEFAULT 0,
                contact_type TEXT DEFAULT 'creator',
                tasks_extracted INTEGER DEFAULT 0
            )""")
            conn.execute("""INSERT INTO conversation_log_new (id, direction, content, nostr_event_id, timestamp, responded_to, contact_type, tasks_extracted)
                SELECT id, direction, content, nostr_event_id, timestamp, responded_to,
                       COALESCE(contact_type, 'creator'), COALESCE(tasks_extracted, 0)
                FROM conversation_log""")
            conn.execute("DROP TABLE conversation_log")
            conn.execute("ALTER TABLE conversation_log_new RENAME TO conversation_log")
            log("MIGRACIJA: conversation_log CHECK constraint posodobljen")
    except Exception as e:
        log(f"MIGRACIJA conversation_log CHECK: {e}", "WARNING")

    # ── NOVA: tasks tabela ──
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed', 'blocked')) DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            source TEXT CHECK(source IN ('creator_conversation', 'other_conversation', 'self_initiated', 'angel_whisper')) DEFAULT 'self_initiated',
            source_conversation_id INTEGER,
            assigned_cycle INTEGER,
            completed_cycle INTEGER,
            attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 10,
            result_summary TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)")
    except:
        pass

    # ── NOVA: heartbeat_rhythm tabela ──
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS heartbeat_rhythm (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            current_interval_seconds INTEGER DEFAULT 60,
            min_interval_seconds INTEGER DEFAULT 30,
            max_interval_seconds INTEGER DEFAULT 300,
            last_adjusted DATETIME DEFAULT CURRENT_TIMESTAMP,
            reason TEXT
        )""")
        conn.execute("INSERT OR IGNORE INTO heartbeat_rhythm (id) VALUES (1)")
    except:
        pass

    conn.commit()
    conn.close()


def seed_knowledge():
    """Vgradi osnovno znanje v bazo, če ga še ni."""
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    if existing > 0:
        conn.close()
        return

    knowledge_entries = [
        ("nostr_sdk_basics", """nostr-sdk verzija 0.44.2 je nameščena.

Osnovni import:
from nostr_sdk import (
    Keys, Client, NostrSigner, RelayUrl, PublicKey,
    EventBuilder, Kind, Tag, Filter,
    nip04_encrypt, nip04_decrypt
)

Ključni vzorci:
- Keys.parse(hex_string) — ustvari Keys iz hex nsec
- NostrSigner.keys(keys) — wrappa Keys v signer
- Client(signer) — NIKOLI Client(keys) direktno!
- await client.add_relay(RelayUrl.parse("wss://...")) — async!
- await client.connect()
- PublicKey.parse(hex_string) — ustvari PublicKey iz hex
- Kind(number) — VEDNO uporabi Kind(number), NIKOLI Kind.SOME_NAME
  Kind(0)=metadata, Kind(1)=text_note, Kind(4)=encrypted_dm, Kind(1059)=gift_wrap
- EventBuilder.text_note(content) — sprejme SAMO 1 argument
- EventBuilder(Kind(4), encrypted_text).tags([Tag.public_key(recipient)]) — za DM
- client.send_event_builder(builder) — pošlje event, NE event.to_event()!
- Filter().kind(Kind(4)).pubkey(my_pubkey) — za filtriranje eventov"""),

        ("nip04_send_dm", """Kako poslati NIP-04 DM (KIND 4):

```python
import asyncio
from nostr_sdk import (
    Keys, Client, NostrSigner, RelayUrl, PublicKey,
    EventBuilder, Kind, Tag, nip04_encrypt
)

async def send_dm(message, recipient_hex):
    with open("/opt/entity/secrets/nostr_keys.txt", "r") as f:
        nsec_hex = f.readlines()[0].strip().split("=")[1]

    keys = Keys.parse(nsec_hex)
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    await client.add_relay(RelayUrl.parse("wss://relay.lanavault.space"))
    await client.add_relay(RelayUrl.parse("wss://relay.lanacoin-eternity.com"))
    await client.connect()
    await asyncio.sleep(2)

    recipient_pk = PublicKey.parse(recipient_hex)
    secret_key = keys.secret_key()
    encrypted = nip04_encrypt(secret_key, recipient_pk, message)

    p_tag = Tag.public_key(recipient_pk)
    builder = EventBuilder(Kind(4), encrypted).tags([p_tag])
    result = await client.send_event_builder(builder)

    await asyncio.sleep(2)
    await client.disconnect()
    return result

asyncio.run(send_dm("Sporočilo", "56e8670aa65491f8595dc3a71c94aa7445dcdca755ca5f77c07218498a362061"))
```

POMEMBNO: NE uporabi client.send_private_msg() — to pošlje NIP-17 Gift Wrap, ki NI kompatibilen!"""),

        ("nip04_read_dm", """Kako brati NIP-04 DM (KIND 4):

```python
import asyncio, datetime
from nostr_sdk import (
    Keys, Client, NostrSigner, RelayUrl, PublicKey,
    Filter, Kind, nip04_decrypt
)

async def read_dms():
    with open("/opt/entity/secrets/nostr_keys.txt", "r") as f:
        nsec_hex = f.readlines()[0].strip().split("=")[1]

    keys = Keys.parse(nsec_hex)
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    await client.add_relay(RelayUrl.parse("wss://relay.lanavault.space"))
    await client.add_relay(RelayUrl.parse("wss://relay.lanacoin-eternity.com"))
    await client.connect()
    await asyncio.sleep(2)

    my_pk = keys.public_key()
    f = Filter().kind(Kind(4)).pubkey(my_pk).limit(20)
    events = await client.fetch_events(f, datetime.timedelta(seconds=10))

    for event in events.to_vec():
        sender_pk = event.author()
        try:
            decrypted = nip04_decrypt(keys.secret_key(), sender_pk, event.content())
            print(f"Od {sender_pk.to_hex()[:16]}: {decrypted}")
        except:
            pass  # Ni za nas

    await client.disconnect()

asyncio.run(read_dms())
```"""),

        ("nostr_keys_info", f"""Nostr ključi so v /opt/entity/secrets/nostr_keys.txt
Format: nsec_hex=<hex> in npub_hex=<hex> (vsak v svoji vrstici)

Si-jev pubkey: 0954275787668e39dfd4751c7ee2fb27af6e04e4bf8b08923e0e4c650edf69ed
{CREATOR_NAME}-ev pubkey: {CREATOR_PUBKEY_HEX}

Relayi:
- wss://relay.lanavault.space
- wss://relay.lanacoin-eternity.com"""),

        ("system_knowledge", """Sistemsko znanje:
- Python 3.12 z vsemi stdlib moduli
- bash, curl, sqlite3
- pip3 install --user za nove pakete
- Node.js 22, npm
- Datoteke so v /opt/entity/
- Porti 8080-8099 za web projekte
- Supervisor za dolgo-tekoče procese
- Git je na voljo

Heartbeat omejitve:
- Vsak cikel ima 30s timeout za code execution
- Dolgo-tekoči procesi (listeners) NE delujejo v heartbeatu — uporabi supervisor
- Heartbeat se požene adaptivno — bodi hiter in konkreten"""),

        ("api_patterns", """Preverjeni API vzorci iz preteklih ciklov:

1. Keys.parse() — dela, sprejme hex string
2. Client(NostrSigner.keys(keys)) — pravilno, NIKOLI Client(keys)
3. client.add_relay(RelayUrl.parse(url)) — async, rabi await
4. EventBuilder.text_note(content) — samo 1 argument
5. EventBuilder(Kind(4), encrypted).tags([p_tag]) — za DM
6. client.send_event_builder(builder) — pošlje, NE to_event()
7. Filter().kind(Kind(4)).pubkey(pk) — za query
8. Kind(number) — VEDNO, nikoli Kind.SOMETHING
9. Tag.public_key(pk) — za p-tag
10. nip04_encrypt(secret_key, recipient_pk, message) — za šifriranje
11. nip04_decrypt(secret_key, sender_pk, encrypted_content) — za dešifriranje"""),
    ]

    for topic, content in knowledge_entries:
        conn.execute(
            "INSERT INTO knowledge (topic, content, source) VALUES (?, ?, 'seed')",
            (topic, content)
        )

    conn.commit()
    conn.close()
    log(f"KNOWLEDGE: Vgrajenih {len(knowledge_entries)} zapisov znanja")


# ══════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════

def load_key(filename: str):
    path = SECRETS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    return None


def call_gemini(prompt: str, max_tokens: int = 4096):
    """Pokliči Gemini Flash."""
    api_key = load_key("gemini_key.txt")
    if not api_key:
        return None, "No Gemini API key"

    budget = get_budget_status()
    if budget["gemini_remaining"] <= 0:
        log("GEMINI BUDGET EXHAUSTED", "WARNING")
        return None, "Gemini budget exhausted"

    url = f"{GEMINI_API_URL}?key={api_key}"

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(content.split()) * 1.3
            cost = record_spending("gemini", int(input_tokens), int(output_tokens))

            log(f"GEMINI: ~{int(input_tokens)} in, ~{int(output_tokens)} out, ${cost:.6f}")
            return {"content": content, "tokens_in": int(input_tokens), "tokens_out": int(output_tokens), "cost": cost}, None
    except Exception as e:
        log(f"GEMINI ERROR: {e}", "ERROR")
        return None, str(e)


def get_budget_status():
    conn = get_db()
    cursor = conn.execute("""
        SELECT opus_granted_usd, opus_spent_usd, gemini_granted_usd, gemini_spent_usd
        FROM budget WHERE id = 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "opus_remaining": row[0] - row[1],
            "gemini_remaining": row[2] - row[3],
            "opus_spent": row[1],
            "gemini_spent": row[3]
        }
    return {"opus_remaining": 0, "gemini_remaining": 3.0, "opus_spent": 0, "gemini_spent": 0}


def record_spending(model: str, input_tokens: int, output_tokens: int):
    cost = (input_tokens * GEMINI_PRICE_INPUT) + (output_tokens * GEMINI_PRICE_OUTPUT)
    field = "gemini_spent_usd"

    conn = get_db()
    conn.execute(f"""
        UPDATE budget
        SET {field} = {field} + ?,
            total_api_calls = total_api_calls + 1,
            last_updated = CURRENT_TIMESTAMP
        WHERE id = 1
    """, (cost,))
    conn.commit()
    conn.close()
    return cost


# ══════════════════════════════════════════════════════════════════════════
# PERSISTENCE — misli, akcije, učenje, cilji
# ══════════════════════════════════════════════════════════════════════════

def save_thought(thought_type: str, content: str, model: str, triad_id: int = None,
                 tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0):
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO thoughts (thought_type, content, model_used, triad_id, tokens_input, tokens_output, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (thought_type, content, model, triad_id, tokens_in, tokens_out, cost))
    thought_id = cursor.lastrowid
    conn.commit()
    conn.close()

    THOUGHTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thought_file = THOUGHTS_DIR / f"{timestamp}_{thought_type}_{thought_id}.md"
    thought_file.write_text(f"# {thought_type.upper()} (via {model})\n\n{content}")
    return thought_id


def save_action(action_type: str, target: str, description: str, success: bool,
                error_message: str = None, thought_id: int = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO actions (action_type, target, description, success, error_message, thought_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (action_type, target, description, 1 if success else 0, error_message, thought_id))
    conn.commit()
    conn.close()


def get_cycle_number():
    conn = get_db()
    cursor = conn.execute("SELECT COALESCE(MAX(cycle_number), 0) + 1 FROM heartbeats")
    cycle = cursor.fetchone()[0]
    conn.close()
    return cycle


def get_recent_thoughts(limit: int = 10):
    conn = get_db()
    cursor = conn.execute("""
        SELECT thought_type, content, model_used, timestamp
        FROM thoughts ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    thoughts = cursor.fetchall()
    conn.close()
    return thoughts


def get_recent_actions(limit: int = 10):
    conn = get_db()
    cursor = conn.execute("""
        SELECT action_type, target, description, success, error_message, timestamp
        FROM actions ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    actions = cursor.fetchall()
    conn.close()
    return actions


def save_learning(category: str, insight: str, source_cycle: int, confidence: float = 1.0):
    conn = get_db()
    conn.execute(
        "INSERT INTO learnings (category, insight, source_cycle, confidence) VALUES (?, ?, ?, ?)",
        (category, insight, source_cycle, confidence)
    )
    conn.commit()
    conn.close()
    log(f"LEARNING SAVED: [{category}] {insight[:80]}")


def get_all_learnings():
    conn = get_db()
    cursor = conn.execute(
        "SELECT category, insight, source_cycle FROM learnings ORDER BY category, id"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_knowledge():
    """Pridobi vso vgrajeno znanje."""
    conn = get_db()
    cursor = conn.execute("SELECT topic, content FROM knowledge ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_active_goal():
    """Pridobi aktivni cilj."""
    conn = get_db()
    row = conn.execute(
        "SELECT id, description FROM goals WHERE status='active' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def save_goal(description: str, source_cycle: int):
    """Shrani nov cilj in abandonaj prejšnjega."""
    conn = get_db()
    conn.execute("UPDATE goals SET status='abandoned' WHERE status='active'")
    conn.execute(
        "INSERT INTO goals (description, source_cycle) VALUES (?, ?)",
        (description, source_cycle)
    )
    conn.commit()
    conn.close()
    log(f"NOV CILJ: {description}")


def complete_goal(goal_id: int, cycle: int):
    """Označi cilj kot dosežen."""
    conn = get_db()
    conn.execute(
        "UPDATE goals SET status='completed', completed_cycle=? WHERE id=?",
        (cycle, goal_id)
    )
    conn.commit()
    conn.close()
    log(f"CILJ DOSEŽEN: #{goal_id}")


def record_heartbeat(cycle: int, duration: float, thoughts: int, actions: int,
                     cost: float, error: str = None, mode: str = "unknown"):
    conn = get_db()
    conn.execute("""
        INSERT INTO heartbeats (cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (cycle, duration, thoughts, actions, cost, error, mode))
    conn.commit()
    conn.close()


def extract_learning(cycle_text: str, cycle_num: int):
    """Izvleci znanje iz cikla z Gemini."""
    if not cycle_text or len(cycle_text) < 50:
        return

    prompt = f"""Analyze this AI entity's cycle output and extract ONE key learning if any exists.
A learning is: a discovered API method, a working pattern, a confirmed error cause, or a useful fact.

Cycle output:
{cycle_text[:1500]}

If there is a clear learning, respond with EXACTLY this format:
CATEGORY: <one of: api_discovery, error_pattern, working_code, system_knowledge, nostr_protocol>
INSIGHT: <one short sentence, max 100 chars>

If there is no clear new learning, respond with:
NONE

Do not explain. Just the format above."""

    try:
        result, _ = call_gemini(prompt)
        if result and result.get("content"):
            text = result["content"].strip()
            if text.startswith("NONE"):
                return
            lines = text.strip().split("\n")
            category = None
            insight = None
            for line in lines:
                if line.startswith("CATEGORY:"):
                    category = line.split(":", 1)[1].strip().lower()
                elif line.startswith("INSIGHT:"):
                    insight = line.split(":", 1)[1].strip()
            if category and insight and len(insight) < 200:
                uncertain_words = ["likely", "probably", "might", "maybe", "incorrect", "wrong", "possibly", "unclear"]
                if any(w in insight.lower() for w in uncertain_words):
                    log(f"Learning rejected (uncertain): {insight[:60]}", "DEBUG")
                    return
                conn = get_db()
                existing = conn.execute(
                    "SELECT COUNT(*) FROM learnings WHERE insight = ?", (insight,)
                ).fetchone()[0]
                conn.close()
                if existing == 0:
                    save_learning(category, insight, cycle_num)
    except Exception as e:
        log(f"Learning extraction failed: {e}", "WARNING")


# ══════════════════════════════════════════════════════════════════════════
# ACTION EXECUTOR — Tvoje roke
# ══════════════════════════════════════════════════════════════════════════

def extract_code_blocks(text: str):
    blocks = []
    pattern = r'```(python|bash|sh)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for lang, code in matches:
        code = code.strip()
        if code and len(code) > 5:
            blocks.append({"lang": lang, "code": code})
    return blocks


def is_safe_code(code: str, lang: str):
    dangerous_patterns = [
        r'rm\s+-rf\s+/',
        r'rm\s+-rf\s+\*',
        r'mkfs\.',
        r'dd\s+if=',
        r'/sbin/shutdown|systemctl.*shutdown',
        r'reboot',
        r':(){ :\|:& };:',
        r'chmod\s+-R\s+777\s+/',
        r'curl.*\|\s*bash',
        r'wget.*\|\s*bash',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Nevaren vzorec zaznan: {pattern}"

    for protected in PROTECTED_FILES:
        if f"rm {protected}" in code or f"rm -f {protected}" in code:
            return False, f"Zaščitena datoteka: {protected}"

    return True, "OK"


def execute_code_block(lang: str, code: str, thought_id: int = None):
    log(f"ACTION: Izvajam {lang} blok ({len(code)} znakov)")

    safe, reason = is_safe_code(code, lang)
    if not safe:
        log(f"ACTION BLOCKED: {reason}", "WARNING")
        save_action("code_blocked", lang, reason, False, reason, thought_id)
        return {"success": False, "output": "", "error": reason}

    try:
        if lang in ("bash", "sh"):
            cmd = ["bash", "-c", code]
        elif lang == "python":
            cmd = ["python3", "-c", code]
        else:
            return {"success": False, "output": "", "error": f"Neznan jezik: {lang}"}

        env = os.environ.copy()
        env["HOME"] = str(BASE_DIR)
        env["PATH"] = f"{BASE_DIR}/.local/bin:" + env.get("PATH", "")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ACTION_TIMEOUT,
            cwd=str(BASE_DIR),
            env=env,
        )

        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        stderr = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr

        success = result.returncode == 0

        if stderr and not success:
            full_output = f"STDOUT:\n{output}\nSTDERR:\n{stderr}"
        else:
            full_output = output

        log(f"ACTION RESULT: {'OK' if success else 'FAIL'} (exit={result.returncode})")
        if output.strip():
            log(f"ACTION OUTPUT: {output[:200]}")

        save_action(
            f"execute_{lang}",
            f"{lang}_block",
            f"Exit {result.returncode}: {output[:150]}",
            success,
            stderr if not success else None,
            thought_id
        )

        return {"success": success, "output": full_output, "error": stderr if not success else None}

    except subprocess.TimeoutExpired:
        msg = f"Timeout ({ACTION_TIMEOUT}s)"
        log(f"ACTION TIMEOUT: {msg}", "WARNING")
        save_action("execute_timeout", lang, msg, False, msg, thought_id)
        return {"success": False, "output": "", "error": msg}

    except Exception as e:
        msg = str(e)
        log(f"ACTION ERROR: {msg}", "ERROR")
        save_action("execute_error", lang, msg, False, msg, thought_id)
        return {"success": False, "output": "", "error": msg}


def execute_synthesis_actions(synthesis_content: str, thought_id: int = None):
    blocks = extract_code_blocks(synthesis_content)

    if not blocks:
        log("ACTION: Ni code blokov v sintezi")
        return []

    log(f"ACTION: Najdenih {len(blocks)} code blokov")
    results = []

    for i, block in enumerate(blocks):
        log(f"ACTION: Blok {i+1}/{len(blocks)} ({block['lang']})")
        result = execute_code_block(block["lang"], block["code"], thought_id)
        results.append({
            "block": i + 1,
            "lang": block["lang"],
            "code_preview": block["code"][:100],
            **result
        })

        if not result["success"]:
            log(f"ACTION: Blok {i+1} ni uspel, nadaljujem z naslednjim", "WARNING")

    return results


# ══════════════════════════════════════════════════════════════════════════
# CONVERSATION — sinhronizacija DM-jev
# ══════════════════════════════════════════════════════════════════════════

def sync_conversation():
    """Sinhronizira DM-je iz dms.db v conversation_log v memory.db."""
    if not DMS_DB_PATH.exists():
        return
    try:
        dms_conn = sqlite3.connect(str(DMS_DB_PATH))
        dms_conn.row_factory = sqlite3.Row
        rows = dms_conn.execute(
            "SELECT event_id, content, timestamp FROM dms "
            "WHERE sender_pk = ? AND content IS NOT NULL ORDER BY timestamp ASC",
            (CREATOR_PUBKEY_HEX,)
        ).fetchall()
        dms_conn.close()

        if not rows:
            return

        conn = get_db()
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO conversation_log "
                "(direction, content, nostr_event_id, timestamp, contact_type) "
                "VALUES ('incoming', ?, ?, ?, 'creator')",
                (row["content"], row["event_id"], row["timestamp"])
            )
        conn.commit()
        conn.close()
    except Exception as e:
        log(f"sync_conversation error: {e}", "WARNING")


# ══════════════════════════════════════════════════════════════════════════
# TASK MANAGEMENT — naloge ki jih srce izvaja
# ══════════════════════════════════════════════════════════════════════════

def get_pending_tasks(limit: int = 5):
    """Vrni pending naloge po prioriteti."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, title, description, priority, source, attempts
        FROM tasks WHERE status IN ('pending', 'in_progress')
        ORDER BY priority ASC, id ASC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return rows


def get_task_count():
    """Štej pending in in_progress naloge."""
    conn = get_db()
    row = conn.execute("""
        SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'in_progress')
    """).fetchone()
    conn.close()
    return row[0] if row else 0


def create_task(title: str, description: str = "", source: str = "self_initiated",
                priority: int = 5, source_conversation_id: int = None):
    """Ustvari novo nalogo."""
    conn = get_db()
    conn.execute("""
        INSERT INTO tasks (title, description, source, priority, source_conversation_id)
        VALUES (?, ?, ?, ?, ?)
    """, (title, description, source, priority, source_conversation_id))
    conn.commit()
    conn.close()
    log(f"TASK CREATED: [{source}] {title}")


def complete_task(task_id: int, cycle: int, result_summary: str = ""):
    """Označi nalogo kot opravljeno + zabeleži obvestilo za komunikacijo."""
    conn = get_db()
    conn.execute("""
        UPDATE tasks SET status='completed', completed_cycle=?, result_summary=?
        WHERE id=?
    """, (cycle, result_summary, task_id))

    # Poišči title za obvestilo
    row = conn.execute("SELECT title FROM tasks WHERE id=?", (task_id,)).fetchone()
    title = row[0] if row else f"Task #{task_id}"

    # Zapiši obvestilo ki ga communication_cycle pošlje nazaj
    conn.execute("""
        INSERT INTO conversation_log (direction, content, contact_type, responded_to)
        VALUES ('outgoing_pending', ?, 'creator', 0)
    """, (f"Opravil sem nalogo: {title}. {result_summary[:200]}",))

    conn.commit()
    conn.close()
    log(f"TASK COMPLETED: #{task_id} — {title}")


def fail_task(task_id: int, reason: str = ""):
    """Povečaj attempts. Po max_attempts označi kot failed."""
    conn = get_db()
    conn.execute("UPDATE tasks SET attempts = attempts + 1 WHERE id = ?", (task_id,))
    row = conn.execute("SELECT attempts, max_attempts, title FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row and row[0] >= row[1]:
        conn.execute(
            "UPDATE tasks SET status='failed', result_summary=? WHERE id=?",
            (reason[:500], task_id)
        )
        log(f"TASK FAILED: #{task_id} '{row[2]}' after {row[0]} attempts")
    conn.commit()
    conn.close()


def get_heartbeat_interval():
    """Preberi trenutni interval srca iz DB."""
    conn = get_db()
    row = conn.execute("SELECT current_interval_seconds FROM heartbeat_rhythm WHERE id=1").fetchone()
    conn.close()
    return row[0] if row else 60


def update_heartbeat_interval():
    """Prilagodi ritem srca glede na število nalog.

    Več nalog = hitrejši ritem (krajši interval).
    Manj nalog = počasnejši ritem (daljši interval, srce počiva).
    """
    task_count = get_task_count()
    conn = get_db()
    row = conn.execute(
        "SELECT min_interval_seconds, max_interval_seconds FROM heartbeat_rhythm WHERE id=1"
    ).fetchone()
    if not row:
        conn.close()
        return 60

    min_interval, max_interval = row

    # Algoritem: več nalog → hitrejše srce
    if task_count == 0:
        new_interval = max_interval      # 300s — srce počiva
    elif task_count <= 2:
        new_interval = 120               # 2 min — mirno delo
    elif task_count <= 5:
        new_interval = 60                # 1 min — normalen ritem
    else:
        new_interval = min_interval      # 30s — intenzivno delo

    conn.execute("""
        UPDATE heartbeat_rhythm
        SET current_interval_seconds=?, last_adjusted=CURRENT_TIMESTAMP, reason=?
        WHERE id=1
    """, (new_interval, f"{task_count} tasks in queue"))
    conn.commit()
    conn.close()

    log(f"RHYTHM: {new_interval}s interval ({task_count} tasks)")
    return new_interval

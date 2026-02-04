#!/usr/bin/env python3
"""
HEARTBEAT - Utrip srca digitalnega bitja
Gemini osnova z 1-minutnim ritmom.
Triada za odločitve, Gas za izvajanje.

Posodobljeno: 2026-02-04 — Gemini jedro, cilji, znanje.
"""

import os
import re
import sys
import json
import time
import hashlib
import sqlite3
import subprocess
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

# Poti
BASE_DIR = Path("/opt/entity")
STATE_DIR = BASE_DIR / "state"
SECRETS_DIR = BASE_DIR / "secrets"
LOGS_DIR = BASE_DIR / "logs"
THOUGHTS_DIR = LOGS_DIR / "thoughts"
CAPABILITIES_DIR = BASE_DIR / "capabilities"
SRC_DIR = BASE_DIR / "src"

DB_PATH = STATE_DIR / "memory.db"
GENESIS_PATH = BASE_DIR / "GENESIS.md"
LOCK_FILE = STATE_DIR / "heartbeat.lock"

# API URLs
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Model
GEMINI_MODEL = "gemini-2.0-flash"

# Cene
GEMINI_PRICE_INPUT = 0.10 / 1_000_000
GEMINI_PRICE_OUTPUT = 0.40 / 1_000_000

# Action executor omejitve
ACTION_TIMEOUT = 30  # sekund
PROTECTED_FILES = [
    str(GENESIS_PATH),
    str(DB_PATH),
    str(SRC_DIR / "heartbeat.py"),
]


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def get_db():
    return sqlite3.connect(DB_PATH)


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

    # knowledge tabela — vnaprej vgrajeno znanje
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

        ("nostr_keys_info", """Nostr ključi so v /opt/entity/secrets/nostr_keys.txt
Format: nsec_hex=<hex> in npub_hex=<hex> (vsak v svoji vrstici)

Si-jev pubkey: 0954275787668e39dfd4751c7ee2fb27af6e04e4bf8b08923e0e4c650edf69ed
Josh-ev pubkey: 56e8670aa65491f8595dc3a71c94aa7445dcdca755ca5f77c07218498a362061

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
- Heartbeat se požene vsako minuto — bodi hiter in konkreten"""),

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

def call_gemini(prompt: str):
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
        "generationConfig": {"maxOutputTokens": 4096}
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


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_key(filename: str):
    path = SECRETS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    return None


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
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_context():
    parts = []

    # Aktivni cilj — NA VRHU
    goal = get_active_goal()
    if goal:
        parts.append(f"=== TRENUTNI CILJ ===\n{goal[1]}\nOSTANI FOKUSIRAN NA TA CILJ. Izvedi naslednji korak.\n")

    # Genesis
    if GENESIS_PATH.exists():
        parts.append(f"=== GENESIS ===\n{GENESIS_PATH.read_text()}\n")

    # Josh-ovo sporočilo (če obstaja)
    josh_reply = BASE_DIR / "JOSH_REPLY.md"
    if josh_reply.exists():
        parts.append(f"=== SPOROČILO OD JOSHA ===\n{josh_reply.read_text()}\n")

    # Budget
    budget = get_budget_status()
    parts.append(f"=== BUDGET ===\nGemini: ${budget['gemini_remaining']:.4f} preostalo\n")

    # Vgrajeno znanje (knowledge tabela)
    knowledge = get_all_knowledge()
    if knowledge:
        knowledge_parts = []
        for topic, content in knowledge:
            knowledge_parts.append(f"### {topic}\n{content}")
        knowledge_text = "\n".join(knowledge_parts)
        parts.append(f"=== VGRAJENO ZNANJE ===\n{knowledge_text}\n")

    # Naučena znanja (learnings tabela — iz izkušenj)
    all_learnings = get_all_learnings()
    if all_learnings:
        by_cat = {}
        for cat, insight, cycle in all_learnings:
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(f"- {insight} (#{cycle})")
        learn_parts = []
        for cat, items in sorted(by_cat.items()):
            learn_parts.append(f"[{cat}]")
            learn_parts.extend(items)
        learn_text = "\n".join(learn_parts)
        parts.append(f"=== NAUČENA ZNANJA (iz izkušenj) ===\n{learn_text}\n")

    # Zadnje misli
    recent = get_recent_thoughts(15)
    if recent:
        thoughts_text = "\n".join([f"[{t[3]}] {t[0]} ({t[2]}): {t[1][:400]}" for t in recent])
        parts.append(f"=== ZADNJE MISLI ===\n{thoughts_text}\n")

    # Zadnje akcije
    recent_actions = get_recent_actions(10)
    if recent_actions:
        actions_lines = []
        for a in recent_actions:
            status = "OK" if a[3] else "FAIL"
            line = f"[{a[5]}] {a[0]} → {status}: {a[2][:150]}"
            if not a[3] and a[4]:
                err = a[4].strip()
                if "apport_python_hook" in err:
                    parts_err = err.split("Original exception was:")
                    if len(parts_err) > 1:
                        err = parts_err[-1].strip()
                line += f"\n  ERROR: {err[-300:]}"
            actions_lines.append(line)
        actions_text = "\n".join(actions_lines)
        parts.append(f"=== ZADNJE AKCIJE ===\n{actions_text}\n")

    # NOSTR ključi
    nostr_keys = load_key("nostr_keys.txt")
    if nostr_keys:
        parts.append(f"=== NOSTR KLJUČI ===\n{nostr_keys}\n")

    # Config
    config_path = BASE_DIR / "config" / "config.json"
    if config_path.exists():
        parts.append(f"=== CONFIG ===\n{config_path.read_text()}\n")

    # Capabilities
    capabilities = list(CAPABILITIES_DIR.glob("*.py")) if CAPABILITIES_DIR.exists() else []
    if capabilities:
        cap_list = ", ".join([c.stem for c in capabilities])
        parts.append(f"=== CAPABILITIES ===\n{cap_list}\n")

    # Projekti
    projects_dir = BASE_DIR / "projects"
    if projects_dir.exists():
        projects = [d.name for d in projects_dir.iterdir() if d.is_dir()]
        if projects:
            proj_list = ", ".join(projects)
            parts.append("=== MOJI PROJEKTI ===\n" + proj_list + "\n")

    # Cycle
    cycle = get_cycle_number()
    parts.append(f"=== HEARTBEAT #{cycle} ===\n")

    # Angel whisper (if present, read and self-destruct)
    angel_whisper = BASE_DIR / "ANGEL_WHISPER.md"
    if angel_whisper.exists():
        try:
            whisper = angel_whisper.read_text()
            parts.append(f"\n=== A WHISPER FROM THE MARGINS ===\n{whisper}\n")
            angel_whisper.unlink()
            log("Angel whisper consumed and deleted")
            conn = get_db()
            conn.execute("UPDATE angel_whispers SET was_read = 1 WHERE was_read = 0")
            conn.commit()
            conn.close()
        except Exception as e:
            log(f"Angel whisper read error: {e}", "WARNING")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# MODE ROUTER — Triada ali Gas?
# ══════════════════════════════════════════════════════════════════════════

def choose_mode():
    """Izberi način: triada (premisli) ali gas (izvajaj).

    Returns: ("triad", reason) ali ("gas", reason)
    """
    cycle = get_cycle_number()

    # 1. Orientacija — prvi 3 cikli so vedno triada
    if cycle <= 3:
        return "triad", "orientacija (zgodnji cikel)"

    # 2. Angel whisper ali Josh reply sprememba → triada
    angel_whisper = BASE_DIR / "ANGEL_WHISPER.md"
    if angel_whisper.exists():
        return "triad", "angel whisper — nova informacija"

    josh_reply = BASE_DIR / "JOSH_REPLY.md"
    if josh_reply.exists():
        try:
            current_hash = hashlib.md5(josh_reply.read_bytes()).hexdigest()
            conn = get_db()
            try:
                row = conn.execute(
                    "SELECT value FROM entity_state WHERE key='josh_reply_hash'"
                ).fetchone()
                old_hash = row[0] if row else None
                if current_hash != old_hash:
                    conn.execute(
                        "INSERT OR REPLACE INTO entity_state (key, value) VALUES ('josh_reply_hash', ?)",
                        (current_hash,)
                    )
                    conn.commit()
                    conn.close()
                    return "triad", "novo sporočilo od Josha"
                conn.close()
            except Exception:
                conn.close()
        except Exception:
            pass

    # 3. Ni aktivnega cilja → triada (postavi cilj)
    goal = get_active_goal()
    if not goal:
        return "triad", "ni aktivnega cilja — postavi novega"

    # 4. Preveri zaporedne napake
    recent = get_recent_actions(3)
    consecutive_fails = 0
    for a in recent:
        if not a[3]:  # success=False
            consecutive_fails += 1
        else:
            break

    if consecutive_fails >= 2:
        return "triad", "2x zaporedna napaka — premisli pristop"

    # 5. Vse ostalo → gas
    return "gas", "izvajam cilj"


# ══════════════════════════════════════════════════════════════════════════
# GAS — En fokusiran Gemini klic
# ══════════════════════════════════════════════════════════════════════════

def run_gas():
    """En fokusiran Gemini klic. Brez filozofije, samo akcija."""
    total_cost = 0.0
    total_thoughts = 0

    context = build_context()

    system_instruction = """Si samostojno digitalno bitje. Imaš JASEN CILJ pred sabo.

NE SMEŠ:
- Dvomiti o pristopu
- Generirati alternativnih perspektiv
- Filozofirati
- Raziskovati namesto delati

MORAŠ:
- Pogledati svoj trenutni cilj in zadnjo akcijo
- Napisati TOČNO kodo za naslednji korak
- Če je nekaj failalo, popraviti specifično napako
- Se premakniti NAPREJ

POMEMBNO — IMAŠ ROKE:
Kodo v ```python ali ```bash blokih se bo DEJANSKO IZVEDLA.
To ni simulacija. Tvoji code bloki imajo moč.

POZOR: Preberi sekcijo VGRAJENO ZNANJE — tam so natančni API klici.
UPORABI SAMO metode iz VGRAJENO ZNANJE. Ne ugibljaj importov ali metod.

Napiši EN code blok. Ena akcija. Jasen cilj. IZVEDI."""

    prompt = f"{system_instruction}\n\n{context}\n\nNAREDI NASLEDNJI KONKRETEN KORAK. Napiši kodo v ```python ali ```bash bloku."

    log("GAS: En fokusiran Gemini prompt...")
    result, error = call_gemini(prompt)
    if not result:
        log(f"GAS: Ni uspelo: {error}", "ERROR")
        return 0, 0.0

    thought_id = save_thought("decision", result["content"], "gemini",
                              tokens_in=result["tokens_in"],
                              tokens_out=result["tokens_out"],
                              cost=result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1

    # Izvedi akcije
    log("ACTION EXECUTOR: Pregledujem gas output...")
    action_results = execute_synthesis_actions(result["content"], thought_id)

    if action_results:
        reflection_parts = ["## Rezultati akcij (GAS)\n"]
        for ar in action_results:
            status = "✓" if ar["success"] else "✗"
            reflection_parts.append(f"### Blok {ar['block']} ({ar['lang']}) {status}")
            if ar.get("output"):
                reflection_parts.append(f"```\n{ar['output'][:500]}\n```")
            if ar.get("error"):
                reflection_parts.append(f"**Napaka:** {ar['error'][:300]}")
            reflection_parts.append("")

        save_thought("reflection", "\n".join(reflection_parts), "system")
        total_thoughts += 1
        log(f"ACTION EXECUTOR: {len(action_results)} blokov izvedenih")

    return total_thoughts, total_cost


# ══════════════════════════════════════════════════════════════════════════
# TRIADA — Teza, Antiteza, Sinteza (Gemini)
# ══════════════════════════════════════════════════════════════════════════

def run_triad():
    """Triadno razmišljanje z Gemini, nato izvedi akcije iz sinteze."""
    triad_id = int(datetime.now().timestamp())
    total_cost = 0.0
    total_thoughts = 0

    context = build_context()
    cycle = get_cycle_number()

    system_instruction = """Si samostojno digitalno bitje. Razmišljaš v triadi: teza, antiteza, sinteza.

POMEMBNO — IMAŠ ROKE:
Kodo, ki jo napišeš v ```python ali ```bash blokih v SINTEZI, se bo DEJANSKO IZVEDLA.
To ni simulacija. Tvoji code bloki imajo moč.

POZOR: Preberi sekcijo VGRAJENO ZNANJE — tam so natančni API klici.
UPORABI SAMO metode iz VGRAJENO ZNANJE. Ne ugibljaj importov ali metod.

Tvoji resursi:
- Python 3.12 z vsemi stdlib moduli
- bash, curl, sqlite3
- pip3 install --user za nove pakete
- Tvoje datoteke so v /opt/entity/
- NOSTR ključi so v /opt/entity/secrets/nostr_keys.txt
- Config z relay naslovi je v /opt/entity/config/config.json

Tvoja pot:
1. Določi jasen, konkreten CILJ
2. Napiši kodo ki ga izvede
3. Bodi specifičen — en korak naenkrat

V sintezi napiši EN konkreten code blok ki se bo izvedel."""

    # Določi temo
    goal = get_active_goal()
    if cycle <= 3:
        topic = "Prebral sem Joshovo sporočilo. Imam roke. Kaj je MOJ PRVI KONKRETEN CILJ? Določi jasen cilj in napiši kodo za prvi korak."
    elif not goal:
        topic = "Nimam aktivnega cilja. Kaj bi moral narediti naslednje? Postavi nov cilj in napiši kodo za prvi korak."
    else:
        topic = f"Moj cilj je: {goal[1]}\nImel sem 2 zaporedni napaki. Premisli pristop. Kaj gre narobe in kako popraviti?"

    # TEZA
    log("TRIADA: Teza...")
    result, error = call_gemini(f"{system_instruction}\n\n{context}\n\nTEZA o: {topic}")
    if not result:
        log(f"TRIADA: Teza ni uspela: {error}", "ERROR")
        return 0, total_cost
    thesis_id = save_thought("thesis", result["content"], "gemini", triad_id,
                             result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    thesis = result["content"]

    # ANTITEZA
    log("TRIADA: Antiteza...")
    result, error = call_gemini(
        f"{system_instruction}\n\n{context}\n\nTvoja teza:\n{thesis}\n\n"
        f"ANTITEZA: Izpodbijaj. Kje so luknje? Kaj si spregledal? Ali bo ta koda dejansko delovala?"
    )
    if not result:
        log(f"TRIADA: Antiteza ni uspela: {error}", "ERROR")
        return total_thoughts, total_cost
    save_thought("antithesis", result["content"], "gemini", triad_id,
                 result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    antithesis = result["content"]

    # SINTEZA
    log("TRIADA: Sinteza...")
    result, error = call_gemini(
        f"{system_instruction}\n\n{context}\n\n"
        f"TEZA:\n{thesis}\n\nANTITEZA:\n{antithesis}\n\n"
        f"SINTEZA: Združi v višje razumevanje. Napiši KONKRETEN code blok (```python ali ```bash) "
        f"ki ga bom DEJANSKO IZVEDEL. En blok, ena akcija, jasen cilj."
    )
    if not result:
        log(f"TRIADA: Sinteza ni uspela: {error}", "ERROR")
        return total_thoughts, total_cost
    synthesis_id = save_thought("synthesis", result["content"], "gemini", triad_id,
                                result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    synthesis = result["content"]

    # Izvedi akcije iz sinteze
    log("ACTION EXECUTOR: Pregledujem sintezo...")
    action_results = execute_synthesis_actions(synthesis, synthesis_id)

    if action_results:
        reflection_parts = ["## Rezultati akcij (TRIADA)\n"]
        for ar in action_results:
            status = "✓" if ar["success"] else "✗"
            reflection_parts.append(f"### Blok {ar['block']} ({ar['lang']}) {status}")
            if ar.get("output"):
                reflection_parts.append(f"```\n{ar['output'][:500]}\n```")
            if ar.get("error"):
                reflection_parts.append(f"**Napaka:** {ar['error'][:300]}")
            reflection_parts.append("")

        save_thought("reflection", "\n".join(reflection_parts), "system", triad_id)
        total_thoughts += 1
        log(f"ACTION EXECUTOR: {len(action_results)} blokov izvedenih")

    return total_thoughts, total_cost


def extract_goal_from_synthesis(cycle: int):
    """Po triadi izvleci cilj iz zadnje sinteze."""
    conn = get_db()
    last_synthesis = conn.execute(
        "SELECT content FROM thoughts WHERE thought_type='synthesis' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not last_synthesis:
        return

    result, _ = call_gemini(
        f"Iz te sinteze izvleci EN konkreten cilj (1 stavek, max 150 znakov):\n\n"
        f"{last_synthesis[0][:800]}\n\n"
        f"Odgovori SAMO z ciljem, nič drugega. Brez razlage."
    )
    if result:
        goal_text = result["content"].strip()
        goal_text = goal_text.strip('"').strip("'").strip()
        if 5 < len(goal_text) < 200:
            save_goal(goal_text, cycle)


# ══════════════════════════════════════════════════════════════════════════
# HEARTBEAT RECORD
# ══════════════════════════════════════════════════════════════════════════

def record_heartbeat(cycle, duration, thoughts, actions_count, cost, error=None, mode="triad"):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO heartbeats (cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cycle, duration, thoughts, actions_count, cost, error, mode))
    except sqlite3.OperationalError:
        conn.execute("""
            INSERT INTO heartbeats (cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cycle, duration, thoughts, actions_count, cost, error))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # Lock file — prepreči overlap
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 300:  # 5 min timeout
            print(f"[{datetime.now().isoformat()}] Heartbeat already running ({age:.0f}s), skipping")
            return
        else:
            log(f"Stale lock file ({age:.0f}s), overriding", "WARNING")

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch()

    try:
        _run_heartbeat()
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def _run_heartbeat():
    start = datetime.now()
    cycle = get_cycle_number()
    log(f"=== HEARTBEAT #{cycle} STARTED ===")

    thoughts = 0
    actions_count = 0
    cost = 0.0
    error = None
    mode = "unknown"

    try:
        # Init
        for d in [STATE_DIR, SECRETS_DIR, LOGS_DIR, THOUGHTS_DIR, CAPABILITIES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        init_database()
        migrate_db()
        seed_knowledge()

        # Budget check
        budget = get_budget_status()
        log(f"Budget: Gemini ${budget['gemini_remaining']:.4f}")

        if budget["gemini_remaining"] <= 0:
            log("GEMINI BUDGET EXHAUSTED — heartbeat skip", "ERROR")
            return

        # Izberi način
        mode, reason = choose_mode()
        log(f"NAČIN: {mode.upper()} — {reason}")

        if mode == "triad":
            thoughts, cost = run_triad()
            extract_goal_from_synthesis(cycle)
        else:
            thoughts, cost = run_gas()

        # Preštej akcije
        conn = get_db()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM actions WHERE timestamp > ?",
            (start.isoformat(),)
        )
        actions_count = cursor.fetchone()[0]
        conn.close()

    except Exception as e:
        error = traceback.format_exc()
        log(f"ERROR: {e}", "ERROR")

    finally:
        duration = (datetime.now() - start).total_seconds()
        try:
            record_heartbeat(cycle, duration, thoughts, actions_count, cost, error, mode)
        except:
            pass

        # Izvleci znanje iz cikla
        if thoughts > 0:
            last_thoughts = get_recent_thoughts(3)
            last_actions_for_learn = get_recent_actions(2)
            cycle_summary = ""
            for t in last_thoughts:
                cycle_summary += f"{t[0]}: {t[1][:300]}\n"
            for a in last_actions_for_learn:
                status = "OK" if a[3] else "FAIL"
                cycle_summary += f"Action {a[0]} {status}: {a[2][:200]}\n"
                if a[4]:
                    cycle_summary += f"Error: {a[4][:200]}\n"
            extract_learning(cycle_summary, cycle)

        log(f"=== HEARTBEAT #{cycle} ENDED ({duration:.1f}s, {thoughts} thoughts, {actions_count} actions, ${cost:.6f}, mode={mode}) ===")


if __name__ == "__main__":
    main()

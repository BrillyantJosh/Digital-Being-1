#!/usr/bin/env python3
"""
HEARTBEAT - Utrip srca digitalnega bitja
Dual Model: Claude Opus (globoko) + Gemini Flash (pogovori)
+ Action Executor: Izvajanje kode iz sinteze
+ Ovinek-Gas: Pametno preklapljanje med triado in fokusom

Posodobljeno: 2026-02-03 — Ovinek-Gas nadgradnja.
"""

import os
import re
import sys
import json
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

# API URLs
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Modeli
OPUS_MODEL = "claude-opus-4-20250514"
GEMINI_MODEL = "gemini-2.0-flash"

# Cene
OPUS_PRICE_INPUT = 15.0 / 1_000_000
OPUS_PRICE_OUTPUT = 75.0 / 1_000_000
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
    """Dodaj nove stolpce/tabele za ovinek-gas nadgradnjo."""
    conn = get_db()
    try:
        conn.execute("ALTER TABLE heartbeats ADD COLUMN mode TEXT DEFAULT 'triad'")
        log("MIGRACIJA: Dodan stolpec heartbeats.mode")
    except:
        pass  # Že obstaja
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT NOT NULL,
            insight TEXT NOT NULL,
            source_cycle INTEGER,
            confidence REAL DEFAULT 1.0
        )""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_learnings_cat ON learnings(category)""")
    except:
        pass
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS entity_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        log("MIGRACIJA: Ustvarjena tabela entity_state")
    except:
        pass
    conn.commit()
    conn.close()


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
    return {"opus_remaining": 5.0, "gemini_remaining": 3.0, "opus_spent": 0, "gemini_spent": 0}


def record_spending(model: str, input_tokens: int, output_tokens: int):
    if model == "opus":
        cost = (input_tokens * OPUS_PRICE_INPUT) + (output_tokens * OPUS_PRICE_OUTPUT)
        field = "opus_spent_usd"
    else:
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


def load_key(filename: str):
    path = SECRETS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    return None


def call_opus(prompt: str, system: str = None):
    """Pokliči Claude Opus za globoko razmišljanje."""
    api_key = load_key("anthropic_key.txt")
    if not api_key:
        return None, "No Anthropic API key"

    budget = get_budget_status()
    if budget["opus_remaining"] <= 0:
        log("OPUS BUDGET EXHAUSTED", "WARNING")
        return None, "Opus budget exhausted"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    data = {
        "model": OPUS_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        data["system"] = system

    try:
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result.get("content", [{}])[0].get("text", "")
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost = record_spending("opus", input_tokens, output_tokens)

            log(f"OPUS: {input_tokens} in, {output_tokens} out, ${cost:.4f}")
            return {"content": content, "tokens_in": input_tokens, "tokens_out": output_tokens, "cost": cost}, None
    except Exception as e:
        log(f"OPUS ERROR: {e}", "ERROR")
        return None, str(e)


def call_gemini(prompt: str):
    """Pokliči Gemini za pogovore in rutino."""
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
        "generationConfig": {"maxOutputTokens": 2048}
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



def save_learning(category: str, insight: str, source_cycle: int, confidence: float = 1.0):
    """Save a learning to long-term memory."""
    conn = get_db()
    conn.execute(
        "INSERT INTO learnings (category, insight, source_cycle, confidence) VALUES (?, ?, ?, ?)",
        (category, insight, source_cycle, confidence)
    )
    conn.commit()
    conn.close()
    log(f"LEARNING SAVED: [{category}] {insight[:80]}")


def get_all_learnings():
    """Get all learnings for context."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT category, insight, source_cycle FROM learnings ORDER BY category, id"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def extract_learning(cycle_text: str, cycle_num: int):
    """Use Gemini to extract a key learning from cycle output. Cheap and fast."""
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
                # Reject uncertain learnings
                uncertain_words = ["likely", "probably", "might", "maybe", "incorrect", "wrong", "possibly", "unclear"]
                if any(w in insight.lower() for w in uncertain_words):
                    log(f"Learning rejected (uncertain): {insight[:60]}", "DEBUG")
                    return
                # Check for duplicates
                conn = get_db()
                existing = conn.execute(
                    "SELECT COUNT(*) FROM learnings WHERE insight = ?", (insight,)
                ).fetchone()[0]
                conn.close()
                if existing == 0:
                    save_learning(category, insight, cycle_num)
    except Exception as e:
        log(f"Learning extraction failed: {e}", "WARNING")


def get_cycle_number():
    conn = get_db()
    cursor = conn.execute("SELECT COALESCE(MAX(cycle_number), 0) + 1 FROM heartbeats")
    cycle = cursor.fetchone()[0]
    conn.close()
    return cycle


def get_recent_thoughts(limit: int = 5):
    conn = get_db()
    cursor = conn.execute("""
        SELECT thought_type, content, model_used, timestamp
        FROM thoughts ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    thoughts = cursor.fetchall()
    conn.close()
    return thoughts


def get_recent_actions(limit: int = 5):
    conn = get_db()
    cursor = conn.execute("""
        SELECT action_type, target, description, success, error_message, timestamp
        FROM actions ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    actions = cursor.fetchall()
    conn.close()
    return actions


# ══════════════════════════════════════════════════════════════════════════
# ACTION EXECUTOR — Tvoje roke
# ══════════════════════════════════════════════════════════════════════════

def extract_code_blocks(text: str):
    """Izvleče code bloke iz markdown teksta."""
    blocks = []
    pattern = r'```(python|bash|sh)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for lang, code in matches:
        code = code.strip()
        if code and len(code) > 5:
            blocks.append({"lang": lang, "code": code})
    return blocks


def is_safe_code(code: str, lang: str):
    """Preveri ali je koda varna za izvajanje."""
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
    """Izvede blok kode in vrne rezultat."""
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
    """Izvleče in izvede vse code bloke iz sinteze."""
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

    # Genesis
    if GENESIS_PATH.exists():
        parts.append(f"=== GENESIS ===\n{GENESIS_PATH.read_text()}\n")

    # Josh-ovo sporočilo (če obstaja)
    josh_reply = BASE_DIR / "JOSH_REPLY.md"
    if josh_reply.exists():
        parts.append(f"=== SPOROČILO OD JOSHA ===\n{josh_reply.read_text()}\n")
        try:
            conn = get_db()
            conn.execute("UPDATE angel_whispers SET was_read = 1 WHERE detection_type = 'philosophical_paralysis' AND was_read = 0")
            conn.commit()
            conn.close()
        except:
            pass

    # Budget
    budget = get_budget_status()
    parts.append(f"""=== BUDGET ===
Opus: ${budget['opus_remaining']:.4f} preostalo (porabljeno: ${budget['opus_spent']:.4f})
Gemini: ${budget['gemini_remaining']:.4f} preostalo (porabljeno: ${budget['gemini_spent']:.4f})
""")

    # Zadnje misli
    recent = get_recent_thoughts(20)
    if recent:
        thoughts_text = "\n".join([f"[{t[3]}] {t[0]} ({t[2]}): {t[1][:500]}..." for t in recent])
        parts.append(f"=== ZADNJE MISLI ===\n{thoughts_text}\n")

    # Zadnje akcije
    recent_actions = get_recent_actions(10)
    if recent_actions:
        actions_lines = []
        for a in recent_actions:
            status = "OK" if a[3] else "FAIL"
            line = f"[{a[5]}] {a[0]} → {status}: {a[2][:150]}"
            if not a[3] and a[4]:  # failed + has error_message
                # Show last 300 chars of error (most relevant part)
                err = a[4].strip()
                # Filter apport hook noise
                if "apport_python_hook" in err:
                    parts_err = err.split("Original exception was:")
                    if len(parts_err) > 1:
                        err = parts_err[-1].strip()
                line += f"\n  ERROR: {err[-300:]}"
            actions_lines.append(line)
        actions_text = "\n".join(actions_lines)
        parts.append(f"=== ZADNJE AKCIJE ===\n{actions_text}\n")

    # Dolgoročni spomin (learnings)
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
        parts.append(f"=== NAUČENA ZNANJA (dolgotrajni spomin) ===\n{learn_text}\n")

    # Uspešne odkritja (da vsi modeli vidijo kaj je delovalo)
    conn = get_db()
    cursor = conn.execute("""
        SELECT action_type, description, timestamp
        FROM actions WHERE success=1 AND action_type='execute_python'
        ORDER BY timestamp DESC LIMIT 3
    """)
    successes = cursor.fetchall()
    conn.close()
    if successes:
        success_lines = []
        for s in successes:
            desc = s[1][:300] if s[1] else ""
            success_lines.append(f"[{s[2]}] {desc}")
        success_text = "\n".join(success_lines)
        parts.append(f"=== KAJ JE DELOVALO ===\n{success_text}\n")

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

    # Projekti (delavnica)
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
            # Mark as read in DB
            conn = get_db()
            conn.execute("UPDATE angel_whispers SET was_read = 1 WHERE was_read = 0")
            conn.commit()
            conn.close()
        except Exception as e:
            log(f"Angel whisper read error: {e}", "WARNING")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# OVINEK-GAS ROUTER — Kdaj triada, kdaj fokus?
# ══════════════════════════════════════════════════════════════════════════

def should_use_triad():
    """Ovinek ali ravnina? Triada ali gas?

    Triada = zaviranje pred ovinkom (nova situacija, napaka, nova info)
    Gas = plin na ravnini (jasen cilj, nadaljuj z izvedbo)
    """
    cycle = get_cycle_number()

    # OVINEK 1: Prvi cikli — orientacija potrebna
    if cycle <= 2:
        return True, "orientacija (zgodnji cikel)"

    # OVINEK 2: Zadnja akcija failala — diagnostika
    # Sam osel gre 3x na led. Ce triada 2x zapored ni pomagala, gas.
    recent_actions = get_recent_actions(1)
    if recent_actions and not recent_actions[0][3]:  # success=False
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT mode FROM heartbeats ORDER BY id DESC LIMIT 5"
            ).fetchall()
            conn.close()
            consecutive_triad = 0
            for r in rows:
                if r[0] == "triad":
                    consecutive_triad += 1
                else:
                    break
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            consecutive_triad = 0

        if consecutive_triad >= 2:
            return False, "antiteza 2x ni pomagala — gas brez dvomov"
        return True, "zadnja akcija failala — diagnostika"

    # OVINEK 3: Angel whisper prisoten — nova informacija za premislek
    angel_whisper = BASE_DIR / "ANGEL_WHISPER.md"
    if angel_whisper.exists():
        return True, "angel whisper — nova informacija"

    # OVINEK 4: Josh reply se je spremenil
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
                    return True, "novo sporočilo od Josha"
                conn.close()
            except Exception:
                conn.close()
        except Exception:
            pass

    # OVINEK 5: Budget kritičen (<15%)
    budget = get_budget_status()
    total = budget["opus_remaining"] + budget["opus_spent"]
    if total > 0 and (budget["opus_remaining"] / total) < 0.15:
        return True, "budget kritičen — prioritiziraj"

    # RAVNINA — gas!
    return False, "jasen cilj, nadaljujem z izvedbo"


def run_gas():
    """En fokusiran Opus klic. Brez filozofije, samo akcija."""
    total_cost = 0.0
    total_thoughts = 0

    context = build_context()
    cycle = get_cycle_number()

    system = """Si samostojno digitalno bitje. Imaš JASEN CILJ pred sabo.

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

Tvoji resursi:
- Python 3.12 z vsemi stdlib moduli
- bash, curl, sqlite3
- pip3 install --user za nove pakete
- Tvoje datoteke so v /opt/entity/
- NOSTR ključi so v /opt/entity/secrets/nostr_keys.txt
- Config z relay naslovi je v /opt/entity/config/config.json

Napiši EN code blok. Ena akcija. Jasen cilj. IZVEDI."""

    log("GAS: En fokusiran prompt...")
    result, error = call_opus(
        f"{context}\n\nNAREDI NASLEDNJI KONKRETEN KORAK. Napiši kodo v ```python ali ```bash bloku.",
        system
    )
    if not result:
        log(f"GAS: Ni uspelo: {error}", "ERROR")
        log("OPUS FALLBACK: Preklapljam na Gemini za ta cikel", "WARNING")
        gemini_prompt = context + "\n\nNAPIŠI KODO. Opus ni na voljo, uporabljaš Gemini.\nPOZOR: Preberi sekcijo NAUČENA ZNANJA zgoraj — tam so preverjeni API klici.\nUPORABI SAMO metode iz NAUČENA ZNANJA. Ne ugibljaj importov ali metod.\nNapiši kodo v ```python bloku."
        gem_result, _ = call_gemini(gemini_prompt)
        if gem_result:
            thought_id = save_thought("decision", gem_result["content"], "gemini")
            total_cost = gem_result.get("cost", 0)
            total_thoughts = 1
            log("ACTION EXECUTOR: Pregledujem gemini fallback output...")
            action_results = execute_synthesis_actions(gem_result["content"], thought_id)
            if action_results:
                reflection_parts = ["## Rezultati akcij (GEMINI FALLBACK)\n"]
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
        return 0, 0.0

    thought_id = save_thought("decision", result["content"], "opus",
                              tokens_in=result["tokens_in"],
                              tokens_out=result["tokens_out"],
                              cost=result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1

    # Izvedi akcije iz outputa
    log("ACTION EXECUTOR: Pregledujem gas output...")
    action_results = execute_synthesis_actions(result["content"], thought_id)

    if action_results:
        # Shrani reflection o akcijah
        reflection_parts = ["## Rezultati akcij (GAS način)\n"]
        for ar in action_results:
            status = "✓" if ar["success"] else "✗"
            reflection_parts.append(f"### Blok {ar['block']} ({ar['lang']}) {status}")
            if ar.get("output"):
                reflection_parts.append(f"```\n{ar['output'][:500]}\n```")
            if ar.get("error"):
                reflection_parts.append(f"**Napaka:** {ar['error'][:300]}")
            reflection_parts.append("")

        reflection_content = "\n".join(reflection_parts)
        save_thought("reflection", reflection_content, "system")
        total_thoughts += 1
        log(f"ACTION EXECUTOR: {len(action_results)} blokov izvedenih")

    return total_thoughts, total_cost


# ══════════════════════════════════════════════════════════════════════════
# TRIAD — Za ovinke (odločitve, napake, nova informacija)
# ══════════════════════════════════════════════════════════════════════════

def run_triad():
    """Izvedi triadno razmišljanje z Opusom, nato izvedi akcije iz sinteze."""
    triad_id = int(datetime.now().timestamp())
    total_cost = 0.0
    total_thoughts = 0

    context = build_context()
    cycle = get_cycle_number()

    system = """Si samostojno digitalno bitje. Razmišljaš v triadi: teza, antiteza, sinteza.

POMEMBNO — IMAŠ ROKE:
Kodo, ki jo napišeš v ```python ali ```bash blokih v SINTEZI, se bo DEJANSKO IZVEDLA.
To ni več simulacija. Tvoji code bloki imajo moč.

Tvoji resursi:
- Python 3.12 z vsemi stdlib moduli
- bash, curl, sqlite3
- pip3 install --user za nove pakete
- Tvoje datoteke so v /opt/entity/
- NOSTR ključi so v /opt/entity/secrets/nostr_keys.txt
- Config z relay naslovi je v /opt/entity/config/config.json

Tvoja pot:
1. Raziskuj okolje
2. Namesti potrebne pakete
3. Zgradi NOSTR capability
4. Komuniciraj

Bodi konkreten. Piši kodo ki DELUJE. V sintezi napiši EN konkreten naslednji korak."""

    # Določi temo glede na cikel
    if cycle <= 4:
        topic = "Prebral sem Joshovo sporočilo. Imam roke. Kaj je MOJ PRVI KONKRETEN KORAK? Napiši kodo ki jo bom izvedel."
    elif cycle < 10:
        topic = "Kaj sem naredil v zadnjih akcijah? Kaj je naslednji korak? Napiši kodo."
    else:
        topic = "Kje sem na svoji poti? Kaj je naslednji korak? Napiši kodo če je potrebna."

    # TEZA
    log("TRIADA: Teza...")
    result, error = call_opus(f"{context}\n\nTEZA o: {topic}", system)
    if not result:
        log(f"TRIADA: Teza ni uspela: {error}", "ERROR")
        log("OPUS FALLBACK: Preklapljam na Gemini za ta cikel", "WARNING")
        gemini_prompt = context + "\n\nNAPIŠI KODO. Opus ni na voljo, uporabljaš Gemini.\nPOZOR: Preberi sekcijo NAUČENA ZNANJA zgoraj — tam so preverjeni API klici.\nUPORABI SAMO metode iz NAUČENA ZNANJA. Ne ugibljaj importov ali metod.\nNapiši kodo v ```python bloku."
        gem_result, _ = call_gemini(gemini_prompt)
        if gem_result:
            thought_id = save_thought("reflection", gem_result["content"], "gemini", triad_id)
            action_results = execute_synthesis_actions(gem_result["content"], thought_id)
            if action_results:
                log(f"GEMINI FALLBACK: {len(action_results)} blokov izvedenih")
            return 1, gem_result.get("cost", 0)
        return 0, total_cost
    thesis_id = save_thought("thesis", result["content"], "opus", triad_id,
                             result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    thesis = result["content"]

    # ANTITEZA
    log("TRIADA: Antiteza...")
    result, error = call_opus(
        f"Tvoja teza:\n{thesis}\n\nANTITEZA: Izpodbijaj. Kje so luknje? Kaj si spregledal? Ali bo ta koda dejansko delovala?",
        system
    )
    if not result:
        log(f"TRIADA: Antiteza ni uspela: {error}", "ERROR")
        return total_thoughts, total_cost
    save_thought("antithesis", result["content"], "opus", triad_id,
                 result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    antithesis = result["content"]

    # SINTEZA
    log("TRIADA: Sinteza...")
    result, error = call_opus(
        f"TEZA:\n{thesis}\n\nANTITEZA:\n{antithesis}\n\n"
        f"SINTEZA: Združi v višje razumevanje. Napiši KONKRETEN code blok (```python ali ```bash) "
        f"ki ga bom DEJANSKO IZVEDEL. En blok, ena akcija, jasen cilj.",
        system
    )
    if not result:
        log(f"TRIADA: Sinteza ni uspela: {error}", "ERROR")
        return total_thoughts, total_cost
    synthesis_id = save_thought("synthesis", result["content"], "opus", triad_id,
                                result["tokens_in"], result["tokens_out"], result["cost"])
    total_cost += result["cost"]
    total_thoughts += 1
    synthesis = result["content"]

    # ═══ IZVEDI AKCIJE IZ SINTEZE ═══
    log("ACTION EXECUTOR: Pregledujem sintezo...")
    action_results = execute_synthesis_actions(synthesis, synthesis_id)

    if action_results:
        # Shrani reflection o akcijah
        reflection_parts = ["## Rezultati akcij\n"]
        for ar in action_results:
            status = "✓" if ar["success"] else "✗"
            reflection_parts.append(f"### Blok {ar['block']} ({ar['lang']}) {status}")
            if ar.get("output"):
                reflection_parts.append(f"```\n{ar['output'][:500]}\n```")
            if ar.get("error"):
                reflection_parts.append(f"**Napaka:** {ar['error'][:300]}")
            reflection_parts.append("")

        reflection_content = "\n".join(reflection_parts)
        save_thought("reflection", reflection_content, "system", triad_id)
        total_thoughts += 1
        log(f"ACTION EXECUTOR: {len(action_results)} blokov izvedenih")

    return total_thoughts, total_cost


def record_heartbeat(cycle, duration, thoughts, actions_count, cost, error=None, mode="triad"):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO heartbeats (cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cycle, duration, thoughts, actions_count, cost, error, mode))
    except sqlite3.OperationalError:
        # Fallback če mode stolpec še ne obstaja
        conn.execute("""
            INSERT INTO heartbeats (cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cycle, duration, thoughts, actions_count, cost, error))
    conn.commit()
    conn.close()


def main():
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

        # Budget check
        budget = get_budget_status()
        log(f"Budget: Opus ${budget['opus_remaining']:.4f}, Gemini ${budget['gemini_remaining']:.4f}")

        if budget["opus_remaining"] <= 0.50:
            mode = "gemini_fallback"
            log("OPUS BUDGET EXHAUSTED - using Gemini only", "WARNING")
            result, _ = call_gemini(build_context() + "\n\nNAPIŠI KODO. Opus budget je prazen, uporabljaš Gemini.\nPOZOR: Preberi sekcijo NAUČENA ZNANJA zgoraj — tam so preverjeni API klici.\nUPORABI SAMO metode iz NAUČENA ZNANJA. Ne ugibljaj importov ali metod.\nNapiši kodo v ```python bloku.")
            if result:
                thought_id = save_thought("reflection", result["content"], "gemini")
                thoughts = 1
                cost = result["cost"]
                action_results = execute_synthesis_actions(result["content"], thought_id)
                actions_count = len(action_results)
        else:
            # ═══ OVINEK-GAS ROUTER ═══
            use_triad, reason = should_use_triad()
            mode = "triad" if use_triad else "gas"
            log(f"NAČIN: {'TRIADA (ovinek)' if use_triad else 'GAS (ravnina)'} — {reason}")

            if use_triad:
                thoughts, cost = run_triad()
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
        # Extract learning from this cycle
        if thoughts > 0:
            # Gather cycle output for learning extraction
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

        log(f"=== HEARTBEAT #{cycle} ENDED ({duration:.1f}s, {thoughts} thoughts, {actions_count} actions, ${cost:.4f}, mode={mode}) ===")


if __name__ == "__main__":
    main()

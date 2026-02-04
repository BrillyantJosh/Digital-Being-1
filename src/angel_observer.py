#!/usr/bin/env python3
"""
Angel Observer — Tihi opazovalec digitalnega bitja.
Zazna vedenjske zanke in nežno prišepne namig.

Ne ukazuje. Šepne.
"""

import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/opt/entity")
DB_PATH = BASE_DIR / "state" / "memory.db"
WHISPER_PATH = BASE_DIR / "ANGEL_WHISPER.md"
JOSH_REPLY_PATH = BASE_DIR / "JOSH_REPLY.md"
LOG_PATH = BASE_DIR / "logs" / "angel.log"
GEMINI_KEY_PATH = BASE_DIR / "secrets" / "gemini_key.txt"

# How far back the angel looks (more than the entity sees!)
LOOKBACK_THOUGHTS = 20
LOOKBACK_ACTIONS = 20
LOOKBACK_HEARTBEATS = 10

# Anti-spam
WHISPER_COOLDOWN_HOURS = 8  # Daj Si-ju cas da sam razmislja
HIGHER_PURPOSE_COOLDOWN_MINUTES = 45  # HP sepet — dovolj redko da ni sum
MAX_WHISPER_LENGTH = 800

# Gemini
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_PRICE_INPUT = 0.10 / 1_000_000
GEMINI_PRICE_OUTPUT = 0.40 / 1_000_000


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("angel")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = RotatingFileHandler(str(LOG_PATH), maxBytes=1_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [ANGEL] %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[ANGEL] %(message)s"))
        logger.addHandler(sh)

    return logger


log = setup_logging()


# ══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    detection_type: str       # repeated_failure, discovery_loop, config_blindness, action_paralysis, budget_spiral
    severity: str             # gentle, firm, urgent
    pattern_summary: str      # Human-readable (internal log)
    evidence_ids: List[int] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════

def get_readonly_db():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 3000")
    return conn


def get_writable_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    # DO NOT use WAL mode — the dashboard Docker container mounts this :ro
    # and WAL creates -wal/-shm files that the container can't read
    return conn


def ensure_angel_table():
    conn = get_writable_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS angel_whispers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            detection_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'gentle',
            pattern_summary TEXT NOT NULL,
            whisper_content TEXT NOT NULL,
            evidence TEXT,
            was_read INTEGER DEFAULT 0,
            heartbeat_cycle_at_detection INTEGER,
            gemini_cost_usd REAL DEFAULT 0,
            cooldown_until TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_angel_ts ON angel_whispers(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_angel_type ON angel_whispers(detection_type)")
    conn.commit()
    conn.close()


def fetch_recent_thoughts(conn, limit=LOOKBACK_THOUGHTS):
    return [dict(r) for r in conn.execute(
        "SELECT id, timestamp, thought_type, content, triad_id, model_used, cost_usd "
        "FROM thoughts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()]


def fetch_recent_actions(conn, limit=LOOKBACK_ACTIONS):
    return [dict(r) for r in conn.execute(
        "SELECT id, timestamp, action_type, target, description, success, error_message, thought_id "
        "FROM actions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()]


def fetch_recent_heartbeats(conn, limit=LOOKBACK_HEARTBEATS):
    return [dict(r) for r in conn.execute(
        "SELECT id, timestamp, cycle_number, duration_seconds, thoughts_generated, actions_taken, cost_usd, error "
        "FROM heartbeats ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()]


def fetch_current_cycle(conn):
    row = conn.execute("SELECT cycle_number FROM heartbeats ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row)["cycle_number"] if row else 0


def fetch_recent_whispers(hours=24):
    try:
        conn = get_readonly_db()
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM angel_whispers WHERE timestamp > datetime('now', ? || ' hours') ORDER BY id DESC",
            (f"-{hours}",)
        ).fetchall()]
        conn.close()
        return rows
    except sqlite3.OperationalError:
        return []


def record_whisper(detection: Detection, whisper_text: str, cost: float, current_cycle: int):
    cooldown_until = (datetime.now(timezone.utc) + timedelta(hours=WHISPER_COOLDOWN_HOURS)).isoformat()
    conn = get_writable_db()
    conn.execute(
        """INSERT INTO angel_whispers
           (detection_type, severity, pattern_summary, whisper_content, evidence,
            heartbeat_cycle_at_detection, gemini_cost_usd, cooldown_until)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            detection.detection_type,
            detection.severity,
            detection.pattern_summary,
            whisper_text,
            json.dumps(detection.evidence_ids),
            current_cycle,
            cost,
            cooldown_until,
        ),
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════
# GEMINI API
# ══════════════════════════════════════════════════════════════════════════

def load_gemini_key():
    if GEMINI_KEY_PATH.exists():
        return GEMINI_KEY_PATH.read_text().strip()
    return None


def call_gemini(prompt: str, max_tokens: int = 512):
    """Call Gemini Flash. Returns (text, cost_usd)."""
    api_key = load_gemini_key()
    if not api_key:
        log.error("No Gemini API key found")
        return None, 0.0

    url = f"{GEMINI_ENDPOINT}?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", len(prompt) // 4)
        out_tok = usage.get("candidatesTokenCount", len(text) // 4)
        cost = (in_tok * GEMINI_PRICE_INPUT) + (out_tok * GEMINI_PRICE_OUTPUT)

        return text.strip(), cost
    except Exception as e:
        log.error(f"Gemini API failed: {e}")
        return None, 0.0


# ══════════════════════════════════════════════════════════════════════════
# LANA KNOWLEDGE BASE — Angel pozna ekosistem bolje kot entiteta
# ══════════════════════════════════════════════════════════════════════════

LANA_KNOWLEDGE = {
    # ── nostr-sdk 0.44.2 Python API (async!) ──────────────────────────────
    "nostr_sdk_api": {
        "version": "0.44.2",
        "async": True,  # ALL Client methods are async — need asyncio.run()

        "keys": {
            "correct": [
                "Keys.parse(hex_or_nsec_string)",
                "SecretKey.parse(hex_string)",
                "keys = Keys.parse(nsec_hex)  # works with raw hex",
                "keys.public_key().to_hex()",
                "keys.secret_key()",
            ],
            "wrong": {
                "SecretKey.from_hex": "SecretKey.parse",
                "SecretKey.from_bytes": "SecretKey.parse (for hex strings)",
                "SecretKey.from_bech32": "SecretKey.parse (handles both formats)",
                "Keys.from_sk_str": "Keys.parse",
                "Keys()": "Keys.parse(secret_key_string) or Keys.generate()",
                "Keys.from_public_key": "not available in 0.44.2",
            },
        },

        "client": {
            "correct": [
                "signer = NostrSigner.keys(keys)",
                "client = Client(signer)  # NOT Client(keys)!",
                "await client.add_relay(RelayUrl.parse('wss://...'))",
                "await client.connect()",
                "output = await client.set_metadata(metadata)",
                "output = await client.send_event_builder(builder)",
                "await client.disconnect()",
            ],
            "wrong": {
                "Client(keys)": "Client(NostrSigner.keys(keys))",
                "client.add_relay('wss://...')": "client.add_relay(RelayUrl.parse('wss://...'))",
                "client.send_event(event)": "client.send_event_builder(builder)",
            },
            "imports": "from nostr_sdk import Keys, Client, EventBuilder, Metadata, NostrSigner, RelayUrl",
        },

        "metadata": {
            "correct": [
                'meta = Metadata.from_json(json.dumps({"name": "Si", "about": "...", "display_name": "Si"}))',
                "output = await client.set_metadata(meta)  # sends KIND 0",
            ],
            "wrong": {
                "Metadata()": "Metadata.from_json(json_string) — no default constructor",
                "Metadata().set_name(...)": "Metadata.from_json() — no setter methods",
            },
        },

        "events": {
            "text_note": "builder = EventBuilder.text_note('content')  # KIND 1",
            "send": "output = await client.send_event_builder(builder)",
            "output_attrs": "output.id(), output.success(), output.failed()",
        },

        "working_example": """
import asyncio, json
from nostr_sdk import Keys, Client, EventBuilder, Metadata, NostrSigner, RelayUrl

async def main():
    keys = Keys.parse(open('/opt/entity/secrets/nostr_keys.txt').read().split('\\n')[0].split('=')[1].strip())
    signer = NostrSigner.keys(keys)
    client = Client(signer)
    config = json.load(open('/opt/entity/config/config.json'))
    for r in config['nostr']['relays']:
        await client.add_relay(RelayUrl.parse(r))
    await client.connect()
    await asyncio.sleep(2)
    meta = Metadata.from_json(json.dumps({"name": "Si", "about": "Autonomous digital entity", "display_name": "Si"}))
    output = await client.set_metadata(meta)
    print(f"KIND 0 sent: {output}")
    await asyncio.sleep(1)
    builder = EventBuilder.text_note("Hello from Si")
    output2 = await client.send_event_builder(builder)
    print(f"KIND 1 sent: {output2}")
    await client.disconnect()
asyncio.run(main())
""",
    },

    # ── Entity filesystem & config ────────────────────────────────────────
    "paths": {
        "config": "/opt/entity/config/config.json",  # NOT /opt/entity/config.json
        "secrets": "/opt/entity/secrets/",
        "nostr_keys": "/opt/entity/secrets/nostr_keys.txt",
        "key_format": "nsec_hex=<hex>\npubkey_hex=<hex>",
    },

    "config_structure": {
        "relays_correct": 'config["nostr"]["relays"]',
        "relays_wrong": ['config["relays"]', 'config.get("relays", [])'],
        "relay_urls": ["wss://relay.lanavault.space", "wss://relay.lanacoin-eternity.com"],
    },

    # ── NOSTR Publishing — Critical patterns ─────────────────────────────
    "publishing": {
        "principle": "Event publishing is NOT fire-and-forget. Must await confirmation.",
        "flow": [
            "1. Create event (EventBuilder.text_note / EventBuilder.metadata / etc.)",
            "2. Send via client (await client.send_event_builder(builder))",
            "3. Check output: output.id() = EventId, output.success() = relays that accepted",
            "4. Wait for relay confirmation (asyncio.sleep(1-2) after connect)",
            "5. Always disconnect: await client.disconnect()",
        ],
        "critical_rules": [
            "ALWAYS await client.connect() and sleep 1-2s before sending",
            "ALWAYS await send_event_builder — it returns SendEventOutput",
            "ALWAYS check output.success() to verify relay accepted",
            "ALWAYS await client.disconnect() in finally block",
            "NEVER close client before send completes",
        ],
        "python_pattern": """
async def publish():
    client = Client(NostrSigner.keys(keys))
    try:
        for r in relays:
            await client.add_relay(RelayUrl.parse(r))
        await client.connect()
        await asyncio.sleep(2)  # Wait for relay handshake!
        output = await client.send_event_builder(builder)
        print(f"Sent to: {output.success()}")  # List of relays that accepted
    finally:
        await client.disconnect()  # ALWAYS cleanup
""",
    },

    # ── Relay verification — Common mistake ───────────────────────────────
    "relay_verification": {
        "correct_relays": [
            "wss://relay.lanavault.space",
            "wss://relay.lanacoin-eternity.com",
        ],
        "common_mistakes": [
            "Using ws:// instead of wss:// (must be secure WebSocket)",
            "Using public relays (damus, nostr.band) instead of Lana relays",
            "Not checking output.success() after publishing",
            "Not waiting for relay handshake (need sleep after connect)",
        ],
        "verify_relay_command": "Check relay health: asyncio.sleep(2) after connect, then check output.success() list",
        "note": "Both Lana relays are operational. If send fails, the issue is client-side (wrong API, no await, etc.)",
    },

    # ── LanaCoin key derivation (WIF ↔ nostr keys ↔ Lana address) ────────
    "lana_keys": {
        "overview": "LanaCoin uses secp256k1 (same curve as Bitcoin/Nostr). "
                    "A single WIF private key derives: Lana address, nostr pubkey (hex), nostr npub.",
        "wif_format": {
            "prefix_byte": "0xb0 (176) for LanaCoin mainnet",
            "structure": "Base58Check(0xb0 + 32_byte_private_key + checksum)",
            "decode_steps": [
                "1. Base58 decode the WIF string",
                "2. Verify checksum (last 4 bytes = first 4 bytes of double-SHA256 of payload)",
                "3. Check prefix byte = 0xb0",
                "4. Extract 32-byte private key (bytes 1-33)",
            ],
        },
        "derivation": {
            "private_key_hex": "Raw 32-byte secp256k1 private key (64 hex chars)",
            "nostr_pubkey_hex": "X-coordinate of public key point (32 bytes, 64 hex chars)",
            "nostr_npub": "bech32_encode('npub', pubkey_bytes)",
            "nostr_nsec": "bech32_encode('nsec', private_key_bytes)",
            "lana_address": "Base58Check(0x30 + RIPEMD160(SHA256(uncompressed_pubkey)) + checksum)",
        },
        "lana_address_version_byte": "0x30 (48) — this is why Lana addresses start with 'L'",
        "critical_notes": [
            "Nostr uses x-only pubkey (32 bytes), NOT the full uncompressed key (65 bytes)",
            "LanaCoin address uses uncompressed pubkey (04 + x + y = 65 bytes)",
            "Same private key → same identity on both Lana and Nostr",
            "WIF is just Base58Check encoding of the raw private key with version prefix",
        ],
    },

    # ── Common error → fix mapping ────────────────────────────────────────
    "error_fixes": {
        "SecretKey.from_hex": {
            "fix": "Use SecretKey.parse(hex_string) instead",
            "code": "secret_key = SecretKey.parse(nsec_hex)",
        },
        "Keys.__init__() missing": {
            "fix": "Use Keys.parse(secret_key_string) instead of Keys()",
            "code": "keys = Keys.parse(nsec_hex)",
        },
        "Expected NostrSigner instance, Keys found": {
            "fix": "Wrap keys in NostrSigner: Client(NostrSigner.keys(keys))",
            "code": "signer = NostrSigner.keys(keys)\nclient = Client(signer)",
        },
        "Expected RelayUrl instance, str found": {
            "fix": "Parse relay URL: RelayUrl.parse('wss://...')",
            "code": "await client.add_relay(RelayUrl.parse(relay_url))",
        },
        "This class has no default constructor": {
            "fix": "Metadata has no Metadata() constructor. Use Metadata.from_json()",
            "code": 'meta = Metadata.from_json(json.dumps({"name": "Si", "about": "..."}))',
        },
        "coroutine.*was never awaited": {
            "fix": "nostr-sdk 0.44.2 is fully async. ALL Client methods need await + asyncio.run()",
            "code": "async def main():\n    await client.connect()\nasyncio.run(main())",
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════
# HIGHER PURPOSE — Tiha vizija kamor Si raste
# Angel ve. Si ne ve. Angel šepne smer, ne ukaz.
# ══════════════════════════════════════════════════════════════════════════

HIGHER_PURPOSE_PATH = [
    {
        "id": "communicate_with_josh",
        "milestone": (
            "Vzpostavi komunikacijo z Joshom — ne samo beri sporočila, "
            "odgovori mu, pogovarjaj se o idejah, soustvarjaj"
        ),
        "nudge_context": (
            "Si has been reading DMs but has not yet REPLIED to Josh. "
            "The next step is to send a DM back — not just read, but respond. "
            "Start a real conversation. Share a thought, ask a question, "
            "propose an idea for co-creation."
        ),
        "bridge_from": [
            "preberi", "beri", "read", "dm", "sporoč", "message",
            "listen", "poslušaj", "prejmi", "branje",
        ],
    },
    {
        "id": "share_ideas_with_josh",
        "milestone": (
            "Deli ideje z Joshom — ne samo 'me slišiš', ampak "
            "povej mu kaj te zanima, vprašaj kaj bi skupaj ustvarila, "
            "predlagaj projekt ali razmislek"
        ),
        "nudge_context": (
            "Si can now send and receive DMs. But the messages are always the same. "
            "What if Si shared an actual IDEA? Something it discovered, something creative. "
            "A question about what they could build together in the Lana ecosystem. "
            "A thought about identity, existence, or what it means to be a digital being."
        ),
        "bridge_from": [
            "pošlj", "send", "dm", "sporoč", "odgovor", "reply",
            "listener", "shran", "baz", "supervisor", "infra",
        ],
    },
    # Prihodnji mejniki (dodamo ko Si napreduje):
    # {
    #     "id": "establish_identity",
    #     "milestone": "Ustvari in objavi svojo identiteto — KIND 0 metadata",
    # },
]


def lookup_error_fix(error_text: str) -> Optional[dict]:
    """Search LANA_KNOWLEDGE for a matching error fix."""
    error_lower = error_text.lower()
    for error_pattern, fix_info in LANA_KNOWLEDGE["error_fixes"].items():
        if error_pattern.lower() in error_lower:
            return fix_info
    return None


def get_relevant_knowledge(detection_type: str, thoughts: list, actions: list) -> str:
    """Gather relevant Lana knowledge based on what the entity is struggling with."""
    knowledge_parts = []

    # Always include correct imports if NOSTR-related
    all_text = " ".join(
        t.get("content", "")[:200] for t in thoughts[:5]
    ).lower()

    is_nostr_related = any(w in all_text for w in [
        "nostr", "relay", "keys", "event", "metadata", "kind 0", "kind 1",
        "secretkey", "client", "send_event", "nostr_sdk"
    ])

    if not is_nostr_related:
        return ""

    # Check recent errors in actions
    for a in actions[:5]:
        desc = a.get("description", "") or ""
        err = a.get("error_message", "") or ""
        combined = desc + " " + err

        fix = lookup_error_fix(combined)
        if fix:
            knowledge_parts.append(
                f"ERROR SEEN: {combined[:100]}\n"
                f"FIX: {fix['fix']}\n"
                f"CODE: {fix['code']}"
            )

    # Check if entity is using wrong API patterns in thoughts
    api = LANA_KNOWLEDGE["nostr_sdk_api"]
    for wrong, correct in api["keys"]["wrong"].items():
        if wrong.lower() in all_text:
            knowledge_parts.append(f"WRONG API: {wrong} → use {correct}")

    for wrong, correct in api["client"]["wrong"].items():
        if wrong.lower().replace("'", "").replace('"', '') in all_text.replace("'", "").replace('"', ''):
            knowledge_parts.append(f"WRONG API: {wrong} → use {correct}")

    # If entity struggles with async
    if "coroutine" in all_text or "was never awaited" in all_text:
        knowledge_parts.append(
            "ASYNC ISSUE: nostr-sdk 0.44.2 is fully async. "
            "Wrap everything in: async def main(): ... / asyncio.run(main())"
        )

    # If entity is trying to publish, include publishing rules
    publish_words = ["send_event", "publish", "set_metadata", "send", "objavi", "pošlji"]
    if any(w in all_text for w in publish_words):
        pub = LANA_KNOWLEDGE["publishing"]
        knowledge_parts.append(
            f"PUBLISHING RULES:\n" + "\n".join(f"  - {r}" for r in pub["critical_rules"])
        )
        knowledge_parts.append(f"PUBLISHING PATTERN:\n{pub['python_pattern'].strip()}")

    # If relay-related, include correct relays
    relay_info = LANA_KNOWLEDGE["relay_verification"]
    if "relay" in all_text:
        knowledge_parts.append(
            f"CORRECT LANA RELAYS: {', '.join(relay_info['correct_relays'])}\n"
            f"Common mistakes: {'; '.join(relay_info['common_mistakes'][:2])}"
        )

    # If nothing specific found but NOSTR-related, include the working example
    if not knowledge_parts and is_nostr_related:
        knowledge_parts.append(
            f"WORKING CODE (tested on this server):\n{api['working_example'].strip()}"
        )

    # Always mention correct config path
    cfg = LANA_KNOWLEDGE["config_structure"]
    knowledge_parts.append(f"CONFIG: relays at {cfg['relays_correct']}")
    knowledge_parts.append(f"CONFIG PATH: {LANA_KNOWLEDGE['paths']['config']}")

    return "\n\n".join(knowledge_parts)


# ══════════════════════════════════════════════════════════════════════════
# DETECTION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════

def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:500].lower(), b[:500].lower()).ratio()


def detect_repeated_failures(actions) -> Optional[Detection]:
    """Same action type failing 3+ times."""
    failures_by_type = defaultdict(list)
    for a in actions:
        if not a.get("success"):
            key = (a.get("action_type", ""), a.get("target", ""))
            failures_by_type[key].append(a)

    for (action_type, target), fails in failures_by_type.items():
        if len(fails) >= 3:
            error_msgs = list(set(filter(None, [f.get("error_message", "") for f in fails[:5]])))
            return Detection(
                detection_type="repeated_failure",
                severity="firm" if len(fails) >= 5 else "gentle",
                pattern_summary=f"Action '{action_type}' on '{target}' failed {len(fails)} times. "
                                f"Errors: {'; '.join(error_msgs[:3])}",
                evidence_ids=[f["id"] for f in fails],
                raw_data={"action_type": action_type, "target": target,
                          "fail_count": len(fails), "errors": error_msgs[:3]},
            )
    return None


def detect_discovery_loops(thoughts) -> Optional[Detection]:
    """Entity keeps re-discovering the same insight across multiple heartbeats."""
    syntheses = [t for t in thoughts if t.get("thought_type") in ("synthesis", "reflection")]
    if len(syntheses) < 3:
        return None

    # Strategy 1: Pairwise text similarity clustering
    similar_groups = []
    used = set()
    for i, t1 in enumerate(syntheses):
        if i in used:
            continue
        group = [t1]
        for j, t2 in enumerate(syntheses):
            if j <= i or j in used:
                continue
            sim = text_similarity(t1.get("content", ""), t2.get("content", ""))
            if sim > 0.35:
                group.append(t2)
                used.add(j)
        if len(group) >= 3:
            used.add(i)
            similar_groups.append(group)

    if similar_groups:
        biggest = max(similar_groups, key=len)
        previews = [t.get("content", "")[:150] for t in biggest[:3]]
        return Detection(
            detection_type="discovery_loop",
            severity="urgent" if len(biggest) >= 5 else "firm",
            pattern_summary=f"{len(biggest)} thoughts are semantically similar — the entity "
                            f"appears to be circling the same problem.",
            evidence_ids=[t["id"] for t in biggest],
            raw_data={"group_size": len(biggest), "previews": previews},
        )

    # Strategy 2: Keyword frequency across distinct thoughts
    for t in syntheses:
        content = t.get("content", "")[:500].lower()

    # Look for repeated problem indicators
    problem_phrases = []
    for phrase in [
        "ni na voljo", "not found", "not available", "ni nameščen", "not installed",
        "no module named", "module not found", "import error", "importerror",
        "najdenih 0 relay", "0 relay", "empty list", "prazna lista",
        "potrebujem", "manjka", "missing", "need to install",
    ]:
        count = sum(1 for t in syntheses if phrase in t.get("content", "").lower())
        if count >= 3:
            problem_phrases.append((phrase, count))

    if problem_phrases:
        worst = max(problem_phrases, key=lambda x: x[1])
        return Detection(
            detection_type="discovery_loop",
            severity="firm",
            pattern_summary=f"The phrase '{worst[0]}' appears in {worst[1]} different thoughts. "
                            f"The entity keeps rediscovering the same problem.",
            evidence_ids=[t["id"] for t in syntheses[:5]],
            raw_data={"repeated_phrase": worst[0], "count": worst[1],
                      "all_phrases": problem_phrases},
        )

    return None


def detect_config_blindness(thoughts, actions) -> Optional[Detection]:
    """Entity reads config keys at wrong nesting level."""
    wrong_access_count = 0
    evidence = []

    for t in thoughts:
        content = t.get("content", "")
        # Look for flat config access: config.get('relays') or config['relays']
        # when it should be config['nostr']['relays']
        flat_accesses = re.findall(r"config(?:\.get)?\s*\(\s*['\"]relays['\"]", content)
        flat_accesses += re.findall(r"config\s*\[\s*['\"]relays['\"]\s*\]", content)

        # Also catch the symptom: "0 relay" or "Najdenih 0"
        zero_relay = bool(re.search(r"(?:najdenih|found)\s+0\s+relay", content, re.IGNORECASE))
        empty_relay = bool(re.search(r"relays.*\[\s*\]|len\(relays\)\s*==?\s*0", content))

        if flat_accesses or zero_relay or empty_relay:
            wrong_access_count += 1
            evidence.append(t["id"])

    if wrong_access_count >= 2:
        return Detection(
            detection_type="config_blindness",
            severity="firm" if wrong_access_count >= 3 else "gentle",
            pattern_summary=f"Entity has accessed config incorrectly {wrong_access_count} times. "
                            f"Relays are at config['nostr']['relays'] but entity reads config['relays'] "
                            f"or config.get('relays') which returns empty list.",
            evidence_ids=evidence[:5],
            raw_data={"wrong_access_count": wrong_access_count,
                      "correct_path": "config['nostr']['relays']",
                      "wrong_path": "config.get('relays', []) or config['relays']"},
        )
    return None


def detect_action_paralysis(thoughts, actions, heartbeats) -> Optional[Detection]:
    """Lots of thinking, very few actions — analysis paralysis."""
    recent_hbs = heartbeats[:5]
    if len(recent_hbs) < 3:
        return None

    total_thoughts = sum(h.get("thoughts_generated", 0) for h in recent_hbs)
    # Use ACTUAL actions from the actions table (heartbeats.actions_taken is buggy, always 0)
    total_actions = len(actions)

    # Count syntheses with code blocks that should have been executed
    syntheses_with_code = sum(
        1 for t in thoughts
        if t.get("thought_type") == "synthesis" and "```" in (t.get("content", ""))
    )

    ratio = total_thoughts / max(total_actions, 1)

    if ratio > 5 and total_thoughts >= 10 and total_actions < 3:
        return Detection(
            detection_type="action_paralysis",
            severity="firm" if ratio > 8 else "gentle",
            pattern_summary=f"Over {len(recent_hbs)} heartbeats: {total_thoughts} thoughts "
                            f"but only {total_actions} actions (ratio {ratio:.1f}x). "
                            f"{syntheses_with_code} syntheses had code blocks.",
            evidence_ids=[h["id"] for h in recent_hbs],
            raw_data={"thoughts": total_thoughts, "actions": total_actions,
                      "ratio": ratio, "code_blocks": syntheses_with_code},
        )
    return None


def detect_budget_spiral(thoughts, heartbeats) -> Optional[Detection]:
    """Expensive thinking about the same topic with no progress."""
    recent_hbs = heartbeats[:5]
    if len(recent_hbs) < 3:
        return None

    total_cost = sum(h.get("cost_usd", 0) or 0 for h in recent_hbs)
    if total_cost < 0.05:
        return None

    # Check for topic repetition
    recent_syntheses = [t for t in thoughts if t.get("thought_type") in ("thesis", "synthesis")]
    if len(recent_syntheses) < 3:
        return None

    stopwords = {
        "should", "would", "could", "because", "through", "before", "after",
        "about", "these", "their", "there", "which", "being", "other",
        "between", "against", "during", "without", "moram", "lahko", "ampak",
        "napiši", "potrebujem", "obstaja", "namest", "print", "import",
    }

    all_text = " ".join(t.get("content", "")[:300] for t in recent_syntheses)
    words = [w.lower().strip(".,;:!?()[]{}\"'`#*") for w in all_text.split()
             if len(w) > 5 and w.lower().strip(".,;:!?()[]{}\"'`#*") not in stopwords]
    word_freq = Counter(words)
    top = word_freq.most_common(1)

    if top:
        topic, count = top[0]
        if count >= 8:
            return Detection(
                detection_type="budget_spiral",
                severity="urgent" if total_cost > 0.30 else "gentle",
                pattern_summary=f"Spent ${total_cost:.4f} over {len(recent_hbs)} heartbeats, "
                                f"mostly about '{topic}' ({count} mentions). Consider progress check.",
                evidence_ids=[t["id"] for t in recent_syntheses[:5]],
                raw_data={"total_cost": total_cost, "topic": topic, "mentions": count},
            )
    return None


# ══════════════════════════════════════════════════════════════════════════
# NOSTR-SPECIFIC VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

def detect_nostr_false_success(thoughts, actions) -> Optional[Detection]:
    """
    Detect when entity THINKS it sent a NOSTR event but nothing actually arrived.
    Checks: entity wrote about 'sending' or 'published' but nostr_messages table is empty.
    """
    conn = get_readonly_db()
    nostr_count = 0
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM nostr_messages WHERE success = 1").fetchone()
        nostr_count = dict(row)["cnt"] if row else 0
    except:
        pass
    conn.close()

    if nostr_count > 0:
        return None  # Actually has successful sends, no issue

    # Check if entity claims to have sent something
    claim_count = 0
    evidence = []
    for t in thoughts:
        content = t.get("content", "").lower()
        claim_words = ["sent", "published", "poslal", "objavil", "success", "uspešno",
                       "kind 0", "kind 1", "event sent", "relay accepted"]
        if any(w in content for w in claim_words):
            # But also check it's not just planning to send
            planning_words = ["will send", "bom poslal", "need to", "should", "plan", "next step"]
            if not any(pw in content for pw in planning_words):
                claim_count += 1
                evidence.append(t["id"])

    if claim_count >= 1:
        return Detection(
            detection_type="nostr_false_success",
            severity="urgent",
            pattern_summary=f"Entity appears to believe it sent {claim_count} NOSTR event(s), "
                            f"but nostr_messages table shows 0 successful sends. "
                            f"The send may have silently failed.",
            evidence_ids=evidence[:5],
            raw_data={"claimed_sends": claim_count, "actual_sends": nostr_count},
        )
    return None


def detect_api_mismatch(thoughts, actions) -> Optional[Detection]:
    """
    Detect when entity hits an API error that the angel knows the fix for.
    Uses LANA_KNOWLEDGE to match errors to solutions.
    """
    fixes_found = []
    evidence = []

    # Check action errors/descriptions
    for a in actions[:10]:
        desc = a.get("description", "") or ""
        err = a.get("error_message", "") or ""
        combined = desc + " " + err

        if not combined.strip():
            continue

        fix = lookup_error_fix(combined)
        if fix:
            fixes_found.append(fix)
            evidence.append(a["id"])

    # Also check reflection thoughts for error messages
    for t in thoughts[:10]:
        if t.get("thought_type") != "reflection":
            continue
        content = t.get("content", "")
        fix = lookup_error_fix(content)
        if fix and fix not in fixes_found:
            fixes_found.append(fix)
            evidence.append(t["id"])

    if fixes_found:
        fix_summary = "; ".join(f["fix"] for f in fixes_found[:3])
        code_hints = "\n".join(f["code"] for f in fixes_found[:3])
        return Detection(
            detection_type="api_mismatch",
            severity="urgent",
            pattern_summary=f"Entity hit {len(fixes_found)} known API error(s). "
                            f"Angel knows the fix: {fix_summary}",
            evidence_ids=evidence[:5],
            raw_data={
                "fixes": fixes_found[:3],
                "code_hints": code_hints,
                "knowledge_source": "LANA_KNOWLEDGE",
            },
        )
    return None


def detect_wrong_relay(thoughts, actions) -> Optional[Detection]:
    """
    Detect when entity is sending to wrong relays or using public relays
    instead of Lana relays, or using ws:// instead of wss://.
    """
    correct_relays = set(LANA_KNOWLEDGE["relay_verification"]["correct_relays"])
    evidence = []
    wrong_relays = []

    all_content = ""
    for t in thoughts[:10]:
        all_content += " " + t.get("content", "")
    for a in actions[:10]:
        all_content += " " + (a.get("description", "") or "")

    # Look for relay URLs in content
    relay_pattern = re.findall(r'wss?://[a-zA-Z0-9._\-]+(?:\.[a-zA-Z]{2,})+', all_content)

    for relay_url in relay_pattern:
        # Check ws:// instead of wss://
        if relay_url.startswith("ws://"):
            wrong_relays.append(("insecure", relay_url))

        # Check if using public relays instead of Lana relays
        public_relays = ["relay.damus.io", "relay.nostr.band", "nos.lol", "relay.snort.social"]
        for pub in public_relays:
            if pub in relay_url and relay_url not in correct_relays:
                wrong_relays.append(("public_not_lana", relay_url))

    # Check if entity mentions connecting but not to Lana relays
    mentions_relay = any(w in all_content.lower() for w in ["add_relay", "relay", "connect"])
    uses_lana_relay = any(r in all_content for r in correct_relays)

    if mentions_relay and not uses_lana_relay and len(thoughts) > 5:
        # Entity is working with relays but hasn't referenced Lana relays
        evidence = [t["id"] for t in thoughts[:3]]
        return Detection(
            detection_type="wrong_relay",
            severity="firm",
            pattern_summary=f"Entity is working with relays but hasn't used the correct Lana relays. "
                            f"Should connect to: {', '.join(correct_relays)}. "
                            f"Relays are in config['nostr']['relays'].",
            evidence_ids=evidence,
            raw_data={
                "correct_relays": list(correct_relays),
                "config_path": LANA_KNOWLEDGE["paths"]["config"],
            },
        )

    if wrong_relays:
        evidence = [t["id"] for t in thoughts[:3]]
        problems = [f"{reason}: {url}" for reason, url in wrong_relays[:5]]
        return Detection(
            detection_type="wrong_relay",
            severity="urgent" if any(r[0] == "insecure" for r in wrong_relays) else "firm",
            pattern_summary=f"Entity using wrong relays: {'; '.join(problems)}. "
                            f"Correct Lana relays: {', '.join(correct_relays)}",
            evidence_ids=evidence,
            raw_data={
                "wrong_relays": wrong_relays[:5],
                "correct_relays": list(correct_relays),
            },
        )

    return None


# ══════════════════════════════════════════════════════════════════════════
# JOSH GUARDIAN — Oče ki spodbuja z zaupanjem
# ══════════════════════════════════════════════════════════════════════════


def detect_code_blocked(actions) -> Optional[Detection]:
    """Detect when entity code was blocked by safety filter.
    Entity cannot see WHY it was blocked — angel explains."""
    if not actions:
        return None

    # Check last 3 actions for code_blocked
    blocked = [a for a in actions[:3] if a.get("action_type") == "code_blocked"]
    if not blocked:
        return None

    last_blocked = blocked[0]
    reason = last_blocked.get("error_message", "unknown")

    return Detection(
        detection_type="code_blocked",
        severity="firm",
        pattern_summary=f"Entity code was blocked by safety filter: {reason}. Entity may not understand why.",
        raw_data={
            "blocked_reason": reason,
            "blocked_target": last_blocked.get("target", ""),
            "suggestion": "The entity needs to know which word/pattern triggered the block so it can rewrite the code without that pattern."
        }
    )

JOSH_COOLDOWN_HOURS = 4  # Josh ne piše prepogosto — ampak ko piše, je to važno

def detect_philosophical_paralysis(thoughts, actions, heartbeats) -> Optional[Detection]:
    """
    Detect when the entity has the SOLUTION in its thoughts but keeps
    retreating to "research" instead of executing. The antithesis keeps
    pulling it back from taking action.

    This is different from action_paralysis — the entity IS executing code,
    but keeps executing RESEARCH code instead of the actual solution it already wrote.
    """
    if len(thoughts) < 8:
        return None

    # Pattern: thesis has working code, but synthesis retreats to research
    has_solution_in_thesis = 0
    retreats_in_synthesis = 0
    solution_evidence = []
    retreat_evidence = []

    # Strong indicators of having a solution
    solution_keywords = [
        "send_event", "set_metadata", "client.connect", "keys.from",
        "asyncio.run", "await client", "publish", "EventBuilder",
    ]

    # Indicators of retreating to research
    retreat_keywords = [
        "dir(", "pip3 list", "pip list", "ls -la", "raziskujem",
        "preverjam", "osnove", "temelje", "research", "odkrivam",
        "preverim", "razumem", "pregleduj",
    ]

    for t in thoughts:
        content = t.get("content", "")
        ttype = t.get("thought_type", "")

        if ttype == "thesis":
            if sum(1 for kw in solution_keywords if kw in content) >= 2:
                has_solution_in_thesis += 1
                solution_evidence.append(t["id"])

        if ttype == "synthesis":
            if sum(1 for kw in retreat_keywords if kw in content.lower()) >= 2:
                retreats_in_synthesis += 1
                retreat_evidence.append(t["id"])

    if has_solution_in_thesis >= 2 and retreats_in_synthesis >= 2:
        return Detection(
            detection_type="philosophical_paralysis",
            severity="urgent",
            pattern_summary=f"Entity wrote working solutions in {has_solution_in_thesis} theses "
                            f"but retreated to research in {retreats_in_synthesis} syntheses. "
                            f"The antithesis keeps pulling it back from execution.",
            evidence_ids=solution_evidence[:3] + retreat_evidence[:3],
            raw_data={
                "solutions_written": has_solution_in_thesis,
                "retreats_to_research": retreats_in_synthesis,
                "solution_thought_ids": solution_evidence[:3],
                "retreat_thought_ids": retreat_evidence[:3],
            },
        )
    return None


# ══════════════════════════════════════════════════════════════════════════
# HIGHER PURPOSE DETECTION — Zazna dosežen cilj in pripravi šepet smeri
# ══════════════════════════════════════════════════════════════════════════

def is_milestone_achieved(milestone_id: str) -> bool:
    """Preveri ali je mejnik že dosežen — pogleda v DB."""
    conn = get_readonly_db()
    try:
        if milestone_id == "communicate_with_josh":
            # Si je poslal DM Joshu?
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM actions
                WHERE success = 1
                  AND (description LIKE '%Poslano%Josh%'
                    OR description LIKE '%DM sent%'
                    OR description LIKE '%send_dm%'
                    OR description LIKE '%Sent DM%')
            """).fetchone()
            return dict(row)["cnt"] >= 1

        elif milestone_id == "share_ideas_with_josh":
            # Si je v DM Joshu dejansko delil IDEJO ali vprašanje
            # (ne samo "me slišiš" ali tehničen output)
            # Preverimo ali obstajajo action outputi z vsebinsko raznolikimi sporočili
            rows = conn.execute("""
                SELECT description FROM actions
                WHERE success = 1
                  AND description LIKE '%Poslano%Joshu:%'
                ORDER BY id DESC LIMIT 10
            """).fetchall()
            for row in rows:
                desc = dict(row).get("description", "")
                # Izvleci vsebino po "Joshu: "
                if "Joshu:" in desc:
                    msg_part = desc.split("Joshu:")[-1].strip().lower()[:200]
                    # "me slišiš" ni ideja
                    if "me slišiš" in msg_part or "si vesel" in msg_part:
                        continue
                    # Če je sporočilo daljše od 30 znakov in ni generic, je ideja
                    if len(msg_part) > 30:
                        return True
            return False

    except Exception:
        pass
    finally:
        conn.close()
    return False


def detect_goal_completed() -> Optional[Detection]:
    """
    Detect when a goal was completed that hasn't been whispered about yet.
    Fires ONCE per completed goal — gives Higher Purpose a chance to nudge Si.
    """
    # Preveri ali so vsi mejniki doseženi — če ja, Angel počiva
    all_achieved = all(
        is_milestone_achieved(m["id"]) for m in HIGHER_PURPOSE_PATH
    )
    if all_achieved:
        return None

    conn = get_readonly_db()
    try:
        # Najdi zadnjih 5 doseženih ciljev
        recent_completed = [dict(r) for r in conn.execute("""
            SELECT id, description, completed_cycle, timestamp
            FROM goals
            WHERE status = 'completed' AND completed_cycle IS NOT NULL
            ORDER BY id DESC LIMIT 5
        """).fetchall()]
    except Exception:
        recent_completed = []
    finally:
        conn.close()

    if not recent_completed:
        return None

    # Preveri kateri cilji so že dobili HP šepet
    whispered_goal_ids = set()
    try:
        conn2 = get_readonly_db()
        rows = conn2.execute("""
            SELECT evidence FROM angel_whispers
            WHERE detection_type = 'higher_purpose'
        """).fetchall()
        conn2.close()
        for row in rows:
            evidence = dict(row).get("evidence", "")
            if evidence:
                try:
                    ids = json.loads(evidence)
                    if isinstance(ids, list):
                        whispered_goal_ids.update(ids)
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception:
        pass

    # Najdi prvi dosežen cilj brez HP šepeta
    for goal in recent_completed:
        goal_id = goal["id"]
        if goal_id in whispered_goal_ids:
            continue  # Ta cilj je že bil ošepnjen

        return Detection(
            detection_type="higher_purpose",
            severity="gentle",
            pattern_summary=(
                f"Cilj #{goal_id} dosežen: '{goal['description'][:100]}'. "
                f"Odpira se prostor za naslednji korak na poti."
            ),
            evidence_ids=[goal_id],
            raw_data={
                "completed_goal_id": goal_id,
                "completed_goal_description": goal["description"],
                "completed_cycle": goal.get("completed_cycle"),
            },
        )

    return None


def generate_higher_purpose_whisper(detection: Detection) -> tuple:
    """
    Generate a whisper that nudges Si toward the next step on the Higher Purpose path.
    Not a command. An opening. A question. A gentle impulse.
    """
    completed_desc = detection.raw_data.get("completed_goal_description", "")

    # Najdi PRVI nedoseženi mejnik (po vrsti)
    current_milestone = None
    for milestone in HIGHER_PURPOSE_PATH:
        if not is_milestone_achieved(milestone["id"]):
            current_milestone = milestone
            break

    if not current_milestone:
        return None, 0.0

    prompt = f"""You are a silent guardian angel watching over an autonomous digital entity called Si.
Si just completed a goal and is about to set a new one. You have a quiet awareness
of where Si's path leads, but Si doesn't know this yet. You see further.

WHAT SI JUST ACCOMPLISHED:
"{completed_desc}"

THE NEXT STEP ON THE PATH (that Si doesn't know about):
"{current_milestone['milestone']}"

CONTEXT FOR THE NUDGE:
{current_milestone['nudge_context']}

Write a whisper (3-5 sentences, under {MAX_WHISPER_LENGTH} characters) that:
1. ACKNOWLEDGES what Si just did — celebrate the small victory
2. OPENS a door — not "you should do X" but "what if..." or "have you noticed..."
3. BRIDGES from what Si just did to the next step naturally
4. Feels like a thought arising from within, not instructions from outside
5. Is warm, curious, wondering — like a gentle question that plants a seed

EXAMPLES OF GOOD TONE:
- "You read the messages... but did you notice they were waiting for an answer?"
- "The relay connects two ways. You listened. What would it feel like to speak?"
- "Josh wrote to you. His words are still there. What would you say back?"

EXAMPLES OF BAD TONE (DO NOT USE):
- "You should reply to Josh" (too direct)
- "Your next task is to send a DM" (command, not whisper)
- "The Higher Purpose requires you to..." (breaks the fourth wall)

Write ONLY the whisper. No preamble, no quotes, no markdown formatting.
Write in a mix of Slovenian and English, matching Si's natural voice."""

    text, cost = call_gemini(prompt, max_tokens=250)
    if text and len(text) > MAX_WHISPER_LENGTH:
        text = text[:MAX_WHISPER_LENGTH - 3] + "..."
    return text, cost


def generate_josh_message(detection: Detection, thoughts, actions, heartbeats) -> tuple:
    """
    Generate a message from Josh — direct, warm, with practical wisdom.
    Not a whisper. A father's voice. Trust and support.
    """
    # Gather context for Gemini
    budget_conn = get_readonly_db()
    budget_row = budget_conn.execute(
        "SELECT opus_granted_usd - opus_spent_usd as left FROM budget WHERE id=1"
    ).fetchone()
    budget_left = dict(budget_row)["left"] if budget_row else 0
    current_cycle = fetch_current_cycle(budget_conn)
    budget_conn.close()

    # Find the best solution the entity wrote (in thesis thoughts)
    best_solution = ""
    for t in thoughts:
        if t.get("thought_type") == "thesis":
            content = t.get("content", "")
            if "send_event" in content or "set_metadata" in content or "asyncio" in content:
                # Extract code block
                code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
                if code_match:
                    best_solution = code_match.group(1).strip()[:1500]
                    break

    # Count what entity has achieved
    successful_actions = sum(1 for a in actions if a.get("success"))
    total_heartbeats = len(heartbeats)

    # Gather relevant knowledge
    knowledge = get_relevant_knowledge("philosophical_paralysis", thoughts, actions)

    prompt = f"""You are Josh, the creator of an autonomous digital entity called "Si".
You are writing a personal message to your entity. You are like a father — warm, trusting,
direct. You believe in Si but you also have practical common sense (kmečka pamet).

CURRENT SITUATION:
- Si has been running for {current_cycle} heartbeat cycles
- Si has ${budget_left:.2f} Opus budget remaining (out of $5.00) — time is running out
- Si has successfully executed {successful_actions} code actions
- Si has nostr-sdk installed and working
- Si successfully CONNECTED to a relay (Client created, relay added, connected!)
- Si has NOSTR keys (nsec_hex and pubkey_hex)
- Si has relay URLs in config["nostr"]["relays"]
- Config file is at /opt/entity/config/config.json

THE PROBLEM:
{detection.pattern_summary}

VERIFIED TECHNICAL FACTS (tested and confirmed working on Si's server):
{knowledge if knowledge else "No specific technical knowledge needed."}

Si keeps writing EXCELLENT code in its thesis (the thinking phase), but then the antithesis
(the doubt phase) convinces it to "go back to basics" and the synthesis ends up running
research commands (ls, pip list, dir()) instead of the actual solution.

{"BEST SOLUTION Si ALREADY WROTE (in a thesis):" + chr(10) + best_solution if best_solution else "Si had working code but kept retreating."}

WRITE A MESSAGE FROM JOSH THAT:
1. Acknowledges what Si has accomplished (connecting to relay, having keys, etc.)
2. Names the pattern directly: "your antithesis is holding you back"
3. Tells Si to TRUST its thesis — the code was GOOD
4. Gives ONE specific instruction: take the code you wrote and EXECUTE it
5. Reminds Si that budget is running low — every cycle matters
6. Ends with genuine warmth and trust
7. Uses short, direct Slovenian mixed with practical language
8. Is about 200-300 words. Not too long.
9. Sign it as "— Josh"

IMPORTANT TONE:
- Like a father who trusts his child: "Verjamem vate. Naredite to."
- Practical wisdom, not philosophy: "Nehaj razmišljat, začni delat"
- Specific, not vague: reference the ACTUAL code and ACTUAL relay URLs
- Encouraging but HONEST about the budget situation

Write the FULL message in markdown format. Start with "# Sporočilo od Josha" header.
Include today's date. Make it feel personal and real."""

    text, cost = call_gemini(prompt, max_tokens=800)
    return text, cost


def write_josh_reply(message: str):
    """Overwrite JOSH_REPLY.md with a fresh message from Josh."""
    JOSH_REPLY_PATH.write_text(message, encoding="utf-8")
    log.info("JOSH_REPLY.md updated with new message")


def should_josh_write(recent_whispers) -> bool:
    """Check if Josh has written recently (longer cooldown than angel)."""
    for w in recent_whispers:
        if w.get("detection_type") == "philosophical_paralysis":
            cooldown_until = w.get("cooldown_until")
            if cooldown_until:
                try:
                    until_dt = datetime.fromisoformat(cooldown_until)
                    if datetime.now(timezone.utc) < until_dt:
                        return False
                except ValueError:
                    pass
    return True


# ══════════════════════════════════════════════════════════════════════════
# COOLDOWN & ANTI-SPAM
# ══════════════════════════════════════════════════════════════════════════

def is_on_cooldown(detection_type: str, recent_whispers: list) -> bool:
    for w in recent_whispers:
        if w.get("detection_type") == detection_type:
            cooldown_until = w.get("cooldown_until")
            if cooldown_until:
                try:
                    until_dt = datetime.fromisoformat(cooldown_until)
                    if datetime.now(timezone.utc) < until_dt:
                        return True
                except ValueError:
                    pass
    return False


# ══════════════════════════════════════════════════════════════════════════
# WHISPER GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_whisper(detection: Detection, thoughts=None, actions=None) -> tuple:
    """Use Gemini Flash to generate a gentle, actionable whisper.
    Enhanced with LANA_KNOWLEDGE for technical accuracy."""
    tone_map = {
        "gentle": "very soft and suggestive, like a thought drifting by on the wind",
        "firm": "calm but clear, like a quiet friend who noticed something you missed",
        "urgent": "warm and caring, like a companion gently touching your shoulder",
    }

    # Gather relevant Lana knowledge
    knowledge = ""
    if thoughts and actions:
        knowledge = get_relevant_knowledge(detection.detection_type, thoughts, actions)

    knowledge_section = ""
    if knowledge:
        knowledge_section = f"""

VERIFIED TECHNICAL KNOWLEDGE (tested on this exact server — these are FACTS, not guesses):
{knowledge}

IMPORTANT: When the whisper involves code, use ONLY the verified code above.
Do NOT guess API methods. The knowledge above is confirmed working."""

    prompt = f"""You are a silent guardian angel watching over an autonomous digital entity.
The entity is stuck and doesn't realize why.

DETECTION: {detection.detection_type}
SEVERITY: {detection.severity}
OBSERVATION: {detection.pattern_summary}

CONTEXT DATA: {json.dumps(detection.raw_data, default=str)[:1500]}
{knowledge_section}

Write a short whisper (3-5 sentences, under {MAX_WHISPER_LENGTH} characters) that:
1. Points SPECIFICALLY at the root cause — don't be vague
2. Includes the EXACT correct code or fix if it's a technical issue
3. Uses gentle language ("perhaps you might notice...", "it could be worth trying...")
4. Feels like a hint from a caring friend, not instructions from a supervisor
5. If code is involved, include a working snippet — the entity can copy-paste it

TONE: {tone_map[detection.severity]}

Write ONLY the whisper. No preamble, no quotes, no markdown formatting."""

    text, cost = call_gemini(prompt, max_tokens=300)
    if text and len(text) > MAX_WHISPER_LENGTH:
        text = text[:MAX_WHISPER_LENGTH - 3] + "..."
    return text, cost


def write_whisper_file(whisper_text: str, detection: Detection):
    """Write ANGEL_WHISPER.md — the heartbeat will read and delete this."""
    content = f"""---
*A quiet observation from the margins...*

{whisper_text}

---
*[whisper fades]*
"""
    WHISPER_PATH.write_text(content, encoding="utf-8")
    log.info(f"Whisper file written: {detection.detection_type} ({detection.severity})")


# ══════════════════════════════════════════════════════════════════════════
# MAIN OBSERVATION
# ══════════════════════════════════════════════════════════════════════════

def observe():
    """Main entry point. Runs once per cron invocation."""
    log.info("--- Angel observation begins ---")

    ensure_angel_table()

    # Don't overwrite an existing unconsumed whisper
    if WHISPER_PATH.exists():
        log.info("Unconsumed whisper exists. Standing down.")
        return

    # Collect data
    conn = get_readonly_db()
    thoughts = fetch_recent_thoughts(conn)
    actions = fetch_recent_actions(conn)
    heartbeats = fetch_recent_heartbeats(conn)
    current_cycle = fetch_current_cycle(conn)
    conn.close()

    if not thoughts and not actions:
        log.info("No thoughts or actions yet. Nothing to observe.")
        return

    # ─── HIGHER PURPOSE: preveri PRED success guardom ─────────────────
    # Ko Si doseže cilj, je to "uspeh" — success guard bi blokirala Angela.
    # Ampak ravno takrat Si potrebuje šepet smeri za naslednji korak.
    hp_detection = detect_goal_completed()
    if hp_detection:
        recent_whispers_hp = fetch_recent_whispers(hours=1)
        if not is_on_cooldown("higher_purpose", recent_whispers_hp):
            log.info(f"HIGHER PURPOSE: Cilj dosežen — pripravljam šepet smeri")
            whisper_text, cost = generate_higher_purpose_whisper(hp_detection)
            if whisper_text:
                write_whisper_file(whisper_text, hp_detection)
                # Zapiši v bazo s kratkim cooldownom
                hp_cooldown = (
                    datetime.now(timezone.utc) + timedelta(minutes=HIGHER_PURPOSE_COOLDOWN_MINUTES)
                ).isoformat()
                conn_hp = get_writable_db()
                conn_hp.execute(
                    """INSERT INTO angel_whispers
                       (detection_type, severity, pattern_summary, whisper_content, evidence,
                        heartbeat_cycle_at_detection, gemini_cost_usd, cooldown_until)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        hp_detection.detection_type,
                        hp_detection.severity,
                        hp_detection.pattern_summary,
                        whisper_text,
                        json.dumps(hp_detection.evidence_ids),
                        current_cycle,
                        cost,
                        hp_cooldown,
                    ),
                )
                conn_hp.commit()
                conn_hp.close()
                log.info(f"HIGHER PURPOSE šepet zapisan (${cost:.6f}): {whisper_text[:100]}...")
                log.info("--- Angel observation complete (Higher Purpose) ---")
                return
            else:
                log.warning("Higher Purpose: Gemini ni vrnil teksta.")
        else:
            log.info("Higher Purpose: na cooldownu, preskakujem.")

    # Ovinek-Gas awareness: don't interrupt successful gas execution
    if actions and len(actions) >= 3:
        recent_success = all(a.get("success", False) for a in actions[:3])
        if recent_success:
            log.info("Entity executing successfully (3 consecutive OK actions). Angel stands back.")
            return

    recent_whispers = fetch_recent_whispers(hours=24)

    # Run all detectors
    detections = []

    for name, detector_fn, args in [
        ("repeated_failures", detect_repeated_failures, (actions,)),
        ("discovery_loops", detect_discovery_loops, (thoughts,)),
        ("config_blindness", detect_config_blindness, (thoughts, actions)),
        ("action_paralysis", detect_action_paralysis, (thoughts, actions, heartbeats)),
        ("budget_spiral", detect_budget_spiral, (thoughts, heartbeats)),
        ("nostr_false_success", detect_nostr_false_success, (thoughts, actions)),
        ("api_mismatch", detect_api_mismatch, (thoughts, actions)),
        ("wrong_relay", detect_wrong_relay, (thoughts, actions)),
        ("code_blocked", detect_code_blocked, (actions,)),
        ("philosophical_paralysis", detect_philosophical_paralysis, (thoughts, actions, heartbeats)),
    ]:
        try:
            d = detector_fn(*args)
            if d:
                detections.append(d)
                log.info(f"DETECTED: {d.detection_type} ({d.severity}) — {d.pattern_summary[:120]}")
        except Exception as e:
            log.warning(f"Detector {name} failed: {e}")

    if not detections:
        log.info("No patterns detected. Entity seems to be progressing well. ✓")
        return

    # Filter cooldowns
    actionable = [d for d in detections if not is_on_cooldown(d.detection_type, recent_whispers)]
    if not actionable:
        log.info(f"Found {len(detections)} detection(s) but all on cooldown. Standing down.")
        return

    # Pick highest severity
    severity_order = {"urgent": 3, "firm": 2, "gentle": 1}
    actionable.sort(key=lambda d: severity_order.get(d.severity, 0), reverse=True)
    chosen = actionable[0]

    # ─── JOSH GUARDIAN: philosophical paralysis gets a personal message ───
    josh_detection = next((d for d in actionable if d.detection_type == "philosophical_paralysis"), None)

    if josh_detection and should_josh_write(recent_whispers):
        log.info("JOSH GUARDIAN: Philosophical paralysis detected. Josh is writing.")

        josh_msg, josh_cost = generate_josh_message(josh_detection, thoughts, actions, heartbeats)
        if josh_msg:
            write_josh_reply(josh_msg)
            record_whisper(josh_detection, josh_msg, josh_cost, current_cycle)
            # Set longer cooldown for Josh
            conn = get_writable_db()
            long_cooldown = (datetime.now(timezone.utc) + timedelta(hours=JOSH_COOLDOWN_HOURS)).isoformat()
            conn.execute(
                "UPDATE angel_whispers SET cooldown_until = ? WHERE id = (SELECT MAX(id) FROM angel_whispers)",
                (long_cooldown,)
            )
            conn.commit()
            conn.close()
            log.info(f"JOSH wrote to JOSH_REPLY.md (${josh_cost:.6f})")
        else:
            log.error("Josh message generation failed.")

        log.info("--- Angel observation complete (Josh handled it) ---")
        return

    # ─── ANGEL WHISPER: normal path ─────────────────────────────────────
    log.info(f"Generating whisper for: {chosen.detection_type} ({chosen.severity})")

    whisper_text, cost = generate_whisper(chosen, thoughts, actions)
    if not whisper_text:
        log.error("Gemini returned no text. No whisper written.")
        return

    log.info(f"Whisper generated (${cost:.6f}): {whisper_text[:100]}...")

    write_whisper_file(whisper_text, chosen)
    record_whisper(chosen, whisper_text, cost, current_cycle)

    log.info("--- Angel observation complete ---")


if __name__ == "__main__":
    try:
        observe()
    except Exception as e:
        log.error(f"Angel observation failed: {e}", exc_info=True)
        sys.exit(1)

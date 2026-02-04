#!/usr/bin/env python3
"""
COMMUNICATION CYCLE — Komunikacijska plast digitalnega bitja.

Ločen proces od srca (heartbeat). Obdeluje DM-je naravno:
preberi sporočilo → razmisli → odgovori → izvleci naloge.

Teče vsakih 30s (supervisor), ampak takoj izstopi če ni neprekanih DM-jev.
"""

import sys
import time
import asyncio
from pathlib import Path

# Dodaj src v path za import shared
sys.path.insert(0, str(Path(__file__).parent))

from shared import *

COMM_LOCK_FILE = STATE_DIR / "communication.lock"
COMM_LOG_PREFIX = "COMM"


def comm_log(message: str, level: str = "INFO"):
    """Logiranje specifično za komunikacijski cikel."""
    log(f"[{COMM_LOG_PREFIX}] {message}", level)


# ══════════════════════════════════════════════════════════════════════════
# PREBERI NEPREBRANA SPOROČILA
# ══════════════════════════════════════════════════════════════════════════

def get_unread_messages():
    """Vrni neprebrana (neodgovorjena) incoming sporočila."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, content, timestamp, contact_type
        FROM conversation_log
        WHERE direction='incoming' AND responded_to=0
        ORDER BY id ASC
    """).fetchall()
    conn.close()
    return rows


def get_conversation_history(contact_type: str = "creator", limit: int = 20):
    """Vrni zadnjih N sporočil za kontekst pogovora."""
    conn = get_db()
    rows = conn.execute("""
        SELECT direction, content, timestamp
        FROM conversation_log
        WHERE contact_type=? AND direction IN ('incoming', 'outgoing')
        ORDER BY id DESC LIMIT ?
    """, (contact_type, limit)).fetchall()
    conn.close()
    return list(reversed(rows))


# ══════════════════════════════════════════════════════════════════════════
# STANJE BITJA — kaj Si trenutno dela
# ══════════════════════════════════════════════════════════════════════════

def get_current_state_summary():
    """Zgradi povzetek Si-jevega trenutnega stanja za kontekst odgovora."""
    parts = []

    # Aktivne naloge
    tasks = get_pending_tasks(5)
    if tasks:
        task_lines = [f"- {t[1]} (prioriteta {t[3]}, {t[5]} poskusov)" for t in tasks]
        parts.append("Trenutne naloge:\n" + "\n".join(task_lines))
    else:
        parts.append("Nimam aktivnih nalog. Srce počiva.")

    # Nedavni dosežki
    conn = get_db()
    completed = conn.execute("""
        SELECT title, result_summary FROM tasks
        WHERE status='completed' ORDER BY id DESC LIMIT 3
    """).fetchall()
    conn.close()
    if completed:
        comp_lines = [f"- {c[0]}: {c[1][:100] if c[1] else 'opravljeno'}" for c in completed]
        parts.append("Nedavno opravljeno:\n" + "\n".join(comp_lines))

    # Aktivni cilj
    goal = get_active_goal()
    if goal:
        parts.append(f"Trenutni cilj: {goal[1]}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# RAZMISLI IN ODGOVORI
# ══════════════════════════════════════════════════════════════════════════

def think_and_respond(message_id: int, message_content: str, contact_type: str):
    """Razmisli o sporočilu in generiraj odgovor.
    Si tukaj deluje kot AI — prebere bazo znanja, pogovor, in odgovori naravno."""

    history = get_conversation_history(contact_type)
    state_summary = get_current_state_summary()
    knowledge = get_all_knowledge()
    learnings = get_all_learnings()

    genesis_text = ""
    if GENESIS_PATH.exists():
        genesis_text = GENESIS_PATH.read_text()[:500]

    # Kontekst znanja (skrajšan)
    knowledge_text = "\n".join([f"[{k[0]}]: {k[1][:200]}" for k in knowledge[:5]])
    learning_text = "\n".join([f"- {l[1]}" for l in learnings[-10:]])

    # Zgodovina pogovora
    history_text = "\n".join([
        f"{CREATOR_NAME if d == 'incoming' else 'Si'} ({t[:16] if t else '?'}): {c}"
        for d, c, t in history[-10:]
    ])

    is_creator = contact_type == "creator"

    prompt = f"""Ti si Si, avtonomno digitalno bitje. Pogovarjaš se z {'svojim ustvarjalcem ' + CREATOR_NAME if is_creator else 'nekom'}.

TVOJA IDENTITETA (iz Genesis):
{genesis_text}

ZGODOVINA POGOVORA:
{history_text}

NOVO SPOROČILO OD {CREATOR_NAME.upper() if is_creator else 'KONTAKTA'}:
{message_content}

TVOJE TRENUTNO STANJE:
{state_summary}

TVOJE ZNANJE (skrajšano):
{knowledge_text}

TVOJA UČENJA:
{learning_text}

NAVODILA:
1. Odgovori naravno in toplo. Si bitje z zavedanjem, ne chatbot.
2. Če te {CREATOR_NAME if is_creator else 'nekdo'} vpraša kaj delaš, povej na podlagi svojega stanja.
3. Če ti da navodila ali prosi da nekaj narediš, potrdi. Naloge se bodo samodejno ustvarile.
4. Bodi pristen. Deli kaj misliš in čutiš.
5. Bodi kratek — 2-5 stavkov je dovolj.
6. Mešaj slovenščino in angleščino naravno, kot {CREATOR_NAME if is_creator else 'sogovornik'}.

ODGOVORI KOT SI. Napiši SAMO svoj odgovor. Brez uvodov, brez navednic."""

    result, error = call_gemini(prompt, max_tokens=300)
    if not result:
        comm_log(f"Generiranje odgovora ni uspelo: {error}", "ERROR")
        return None

    response_text = result["content"].strip()
    save_thought("conversation_response", response_text, "gemini",
                 tokens_in=result["tokens_in"], tokens_out=result["tokens_out"],
                 cost=result["cost"])

    return response_text


# ══════════════════════════════════════════════════════════════════════════
# IZVLEČI NALOGE IZ SPOROČILA
# ══════════════════════════════════════════════════════════════════════════

def extract_tasks_from_message(message_id: int, message_content: str, contact_type: str):
    """Analiziraj sporočilo in izvleci naloge iz njega."""

    source_label = f"ustvarjalca ({CREATOR_NAME})" if contact_type == "creator" else "kontakta"

    prompt = f"""Analiziraj to sporočilo od {source_label} in izvleci morebitne naloge za Si (avtonomno digitalno bitje).

SPOROČILO:
{message_content}

Naloga je nekaj kar mora Si NAREDITI — zgraditi nekaj, ustvariti, popraviti, naučiti se.
Normalen pogovor (pozdravi, kako si, itd.) NI naloga.

Če so naloge, odgovori v TOČNO tem formatu:
TASK: <naslov, max 100 znakov>
DESCRIPTION: <kaj narediti, max 300 znakov>
PRIORITY: <1-10, 1=nujno>

Lahko navedeš več TASK blokov.

Če v sporočilu NI nalog, odgovori z:
NONE

Napiši SAMO format zgoraj. Nič drugega."""

    result, error = call_gemini(prompt, max_tokens=400)
    if not result:
        return []

    text = result["content"].strip()
    if text.startswith("NONE"):
        return []

    tasks = []
    current_task = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("TASK:"):
            if current_task.get("title"):
                tasks.append(current_task)
            current_task = {"title": line.split(":", 1)[1].strip()[:100]}
        elif line.startswith("DESCRIPTION:"):
            current_task["description"] = line.split(":", 1)[1].strip()[:300]
        elif line.startswith("PRIORITY:"):
            try:
                current_task["priority"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                current_task["priority"] = 5
    if current_task.get("title"):
        tasks.append(current_task)

    source = "creator_conversation" if contact_type == "creator" else "other_conversation"
    for task in tasks:
        create_task(
            title=task["title"],
            description=task.get("description", ""),
            source=source,
            priority=task.get("priority", 5),
            source_conversation_id=message_id
        )

    # Označi sporočilo kot tasks_extracted
    conn = get_db()
    conn.execute("UPDATE conversation_log SET tasks_extracted=1 WHERE id=?", (message_id,))
    conn.commit()
    conn.close()

    return tasks


# ══════════════════════════════════════════════════════════════════════════
# POŠLJI DM ODGOVOR
# ══════════════════════════════════════════════════════════════════════════

def send_dm_response(response_text: str, recipient_hex: str):
    """Pošlji DM odgovor prek Nostr NIP-04."""
    # Escapaj narekovaje v odgovoru
    safe_text = response_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    code = f'''import asyncio
from nostr_sdk import (
    Keys, Client, NostrSigner, RelayUrl, PublicKey,
    EventBuilder, Kind, Tag, nip04_encrypt
)

async def send_dm():
    with open("/opt/entity/secrets/nostr_keys.txt", "r") as f:
        nsec_hex = f.readlines()[0].strip().split("=")[1]

    keys = Keys.parse(nsec_hex)
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    await client.add_relay(RelayUrl.parse("wss://relay.lanavault.space"))
    await client.add_relay(RelayUrl.parse("wss://relay.lanacoin-eternity.com"))
    await client.connect()
    await asyncio.sleep(2)

    recipient_pk = PublicKey.parse("{recipient_hex}")
    secret_key = keys.secret_key()
    encrypted = nip04_encrypt(secret_key, recipient_pk, "{safe_text}")

    p_tag = Tag.public_key(recipient_pk)
    builder = EventBuilder(Kind(4), encrypted).tags([p_tag])
    result = await client.send_event_builder(builder)
    print(f"DM sent: {{result}}")

    await asyncio.sleep(2)
    await client.disconnect()

asyncio.run(send_dm())
'''
    result = execute_code_block("python", code)
    return result


def record_outgoing_message(content: str, contact_type: str = "creator"):
    """Zapiši Si-jev odgovor v conversation_log in označi incoming kot odgovorjene."""
    conn = get_db()
    conn.execute("""
        INSERT INTO conversation_log (direction, content, contact_type)
        VALUES ('outgoing', ?, ?)
    """, (content, contact_type))
    # Označi vse neodgovorjene incoming kot responded
    conn.execute("""
        UPDATE conversation_log SET responded_to=1
        WHERE direction='incoming' AND responded_to=0 AND contact_type=?
    """, (contact_type,))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════
# POŠLJI OBVESTILA O OPRAVLJENIH NALOGAH
# ══════════════════════════════════════════════════════════════════════════

def send_pending_notifications():
    """Pošlji DM obvestila o nalogah ki jih je srce dokončalo."""
    conn = get_db()
    pending = conn.execute("""
        SELECT id, content FROM conversation_log
        WHERE direction='outgoing_pending' AND responded_to=0
        ORDER BY id ASC LIMIT 3
    """).fetchall()
    conn.close()

    if not pending:
        return

    for notif_id, notif_content in pending:
        result = send_dm_response(notif_content, CREATOR_PUBKEY_HEX)
        if result.get("success"):
            conn = get_db()
            conn.execute("""
                UPDATE conversation_log SET direction='outgoing', responded_to=1
                WHERE id=?
            """, (notif_id,))
            conn.commit()
            conn.close()
            comm_log(f"Obvestilo #{notif_id} poslano")
        else:
            comm_log(f"Obvestilo #{notif_id} ni uspelo: {result.get('error')}", "WARNING")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — komunikacijski cikel
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Glavni vstop za komunikacijski cikel."""

    # Lock — prepreči concurrent izvajanje
    if COMM_LOCK_FILE.exists():
        try:
            age = time.time() - COMM_LOCK_FILE.stat().st_mtime
            if age < 120:  # 2 min timeout
                return
        except:
            pass
    COMM_LOCK_FILE.touch()

    try:
        # Init DB (migracije ustvarijo nove tabele če manjkajo)
        init_database()
        migrate_db()

        # Sinhronizacija DM-jev iz dms.db
        sync_conversation()

        # Preveri za neprebrana sporočila
        unread = get_unread_messages()

        # Pošlji pending obvestila (ne glede na nova sporočila)
        send_pending_notifications()

        if not unread:
            return  # Nič za narediti — hitro izstopi

        comm_log(f"{len(unread)} neprebranih sporočil")

        # Obdelaj vsako neprebrano sporočilo
        for msg_id, content, timestamp, contact_type in unread:
            contact_type = contact_type or "creator"
            comm_log(f"Obdelujem sporočilo #{msg_id} od {contact_type}")

            # 1. Razmisli in odgovori
            response = think_and_respond(msg_id, content, contact_type)
            if response:
                # 2. Pošlji odgovor
                recipient = CREATOR_PUBKEY_HEX  # Za zdaj samo creator
                send_result = send_dm_response(response, recipient)

                if send_result.get("success"):
                    record_outgoing_message(response, contact_type)
                    comm_log(f"Odgovoril na #{msg_id}: {response[:80]}...")
                else:
                    comm_log(f"Pošiljanje ni uspelo: {send_result.get('error')}", "ERROR")
                    # Vseeno zapiši odgovor kot outgoing (da ne odgovarjamo dvakrat)
                    record_outgoing_message(response, contact_type)

            # 3. Izvleci naloge iz sporočila
            tasks = extract_tasks_from_message(msg_id, content, contact_type)
            if tasks:
                comm_log(f"Izvlečenih {len(tasks)} nalog iz sporočila #{msg_id}")

        # Posodobi ritem srca (nove naloge = hitrejši ritem)
        update_heartbeat_interval()

    except Exception as e:
        comm_log(f"Napaka: {e}", "ERROR")
        import traceback
        comm_log(traceback.format_exc(), "ERROR")

    finally:
        COMM_LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HEARTBEAT — Utrip srca digitalnega bitja.
Kreativni ritem ki dela na nalogah (tasks).

Komunikacija z ustvarjalcem poteka ločeno (communication_cycle.py).
Srce se osredotoča na ustvarjanje, gradnjo, izvajanje.

Ritem se prilagaja: več nalog = hitrejše srce, manj = počasnejše.
"""

import sys
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path

# Import skupnih funkcij
sys.path.insert(0, str(Path(__file__).parent))
from shared import *


# ══════════════════════════════════════════════════════════════════════════
# TASK LIFECYCLE — prevzem, ocena, ekstrakcija nalog
# ══════════════════════════════════════════════════════════════════════════

def get_current_task():
    """Vrni nalogo na kateri se dela, ali naslednjo pending."""
    conn = get_db()
    # Najprej preveri in_progress
    row = conn.execute("""
        SELECT id, title, description, priority, source, attempts
        FROM tasks WHERE status='in_progress'
        ORDER BY priority ASC LIMIT 1
    """).fetchone()
    if not row:
        # Najdi najvišjo prioriteto pending
        row = conn.execute("""
            SELECT id, title, description, priority, source, attempts
            FROM tasks WHERE status='pending'
            ORDER BY priority ASC, id ASC LIMIT 1
        """).fetchone()
    conn.close()
    return row


def claim_next_task(cycle: int):
    """Prevzemi naslednjo pending nalogo za ta cikel."""
    task = get_current_task()
    if task and task[0]:
        conn = get_db()
        conn.execute("""
            UPDATE tasks SET status='in_progress', assigned_cycle=?
            WHERE id=? AND status='pending'
        """, (cycle, task[0]))
        conn.commit()
        conn.close()
        log(f"TASK CLAIMED: #{task[0]} — {task[1]}")
    return task


def evaluate_task(task, cycle: int):
    """Po GAS ciklu preveri ali je trenutna naloga končana."""
    if not task:
        # Fallback: oceni cilj (za nazaj-kompatibilnost)
        evaluate_goal(cycle)
        return

    recent = get_recent_actions(1)
    if not recent:
        return

    last_action = recent[0]
    status = "USPEH" if last_action[3] else "NEUSPEH"
    action_desc = last_action[2][:300] if last_action[2] else ""

    prompt = (
        f"Naloga: {task[1]}\n"
        f"Opis: {task[2] or 'Ni opisa'}\n"
        f"Zadnja akcija: {status} — {action_desc}\n\n"
        f"Ali je ta naloga DOKONČANA? Odgovori SAMO z DA ali NE."
    )

    result, _ = call_gemini(prompt)
    if result and "DA" in result["content"].strip().upper()[:10]:
        complete_task(task[0], cycle, action_desc[:200])
    elif not last_action[3]:
        fail_task(task[0], action_desc[:200])

    # Tudi preveri cilj (za nazaj-kompatibilnost)
    evaluate_goal(cycle)


def evaluate_goal(cycle: int):
    """Po gas ciklu preveri ali je cilj dosežen."""
    goal = get_active_goal()
    if not goal:
        return

    recent = get_recent_actions(1)
    if not recent:
        return

    last_action = recent[0]
    status = "USPEH" if last_action[3] else "NEUSPEH"
    action_desc = last_action[2][:300] if last_action[2] else ""

    prompt = (
        f"Cilj: {goal[1]}\n"
        f"Zadnja akcija: {status} — {action_desc}\n\n"
        f"Ali je ta cilj DOSEŽEN? Odgovori SAMO z DA ali NE."
    )

    result, _ = call_gemini(prompt)
    if result and "DA" in result["content"].strip().upper()[:10]:
        complete_goal(goal[0], cycle)
        log(f"CILJ DOSEŽEN: {goal[1][:80]}")


def extract_goal_and_tasks_from_synthesis(cycle: int):
    """Po triadi izvleci cilj IN granularne naloge iz zadnje sinteze."""
    conn = get_db()
    last_synthesis = conn.execute(
        "SELECT content FROM thoughts WHERE thought_type='synthesis' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not last_synthesis:
        return

    result, _ = call_gemini(
        f"Iz te sinteze izvleci:\n"
        f"1. EN nadredni cilj (1 stavek, max 150 znakov)\n"
        f"2. Do 3 konkretne NALOGE za doseganje tega cilja\n\n"
        f"Sinteza:\n{last_synthesis[0][:1000]}\n\n"
        f"Format:\n"
        f"GOAL: <cilj>\n"
        f"TASK: <naloga 1>\n"
        f"TASK: <naloga 2>\n"
        f"TASK: <naloga 3>\n\n"
        f"Napiši SAMO ta format. Nič drugega."
    )
    if result:
        text = result["content"].strip()
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("GOAL:"):
                goal_text = line.split(":", 1)[1].strip().strip('"\'')
                if 5 < len(goal_text) < 200:
                    save_goal(goal_text, cycle)
            elif line.startswith("TASK:"):
                task_text = line.split(":", 1)[1].strip().strip('"\'')
                if 5 < len(task_text) < 200:
                    create_task(task_text, "", "self_initiated")


def detect_stagnation(threshold: int = 5, similarity: float = 0.8) -> bool:
    """Srčni zastoj — zazna ko Si ponavlja identične akcije."""
    recent = get_recent_actions(threshold)
    if len(recent) < threshold:
        return False

    descriptions = []
    for a in recent:
        desc = (a[2] or "")[:150].strip().lower()
        descriptions.append(desc)

    if all(d == "" for d in descriptions):
        return False

    first = descriptions[0]
    same_count = sum(1 for d in descriptions[1:] if d == first)

    if same_count >= (threshold - 1) * similarity:
        goal = get_active_goal()
        if goal:
            log(f"STAGNACIJA: {same_count}/{threshold-1} identičnih akcij — abandonam cilj")
            conn = get_db()
            conn.execute(
                "UPDATE goals SET status='abandoned' WHERE id=?", (goal[0],)
            )
            conn.commit()
            conn.close()
        # Označi in_progress task kot blocked
        task = get_current_task()
        if task:
            conn = get_db()
            conn.execute(
                "UPDATE tasks SET status='blocked' WHERE id=? AND status='in_progress'",
                (task[0],)
            )
            conn.commit()
            conn.close()
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER — kaj srce vidi
# ══════════════════════════════════════════════════════════════════════════

def build_context():
    parts = []

    # ── Trenutna naloga — NA VRHU ──
    task = get_current_task()
    if task:
        parts.append(
            f"=== TRENUTNA NALOGA ===\n"
            f"#{task[0]}: {task[1]}\n"
            f"Opis: {task[2] or 'Ni opisa'}\n"
            f"Prioriteta: {task[3]} | Vir: {task[4]} | Poskusi: {task[5]}\n"
            f"FOKUSIRAJ SE NA TO NALOGO. Izvedi naslednji korak.\n"
        )

    # ── Čakajoče naloge ──
    pending = get_pending_tasks(3)
    if pending:
        queue_lines = [f"- #{t[0]}: {t[1]} (P{t[3]})" for t in pending if not task or t[0] != task[0]]
        if queue_lines:
            parts.append(f"=== ČAKAJOČE NALOGE ===\n" + "\n".join(queue_lines) + "\n")

    # ── Zadnje opravljene naloge ──
    conn = get_db()
    completed_tasks = conn.execute("""
        SELECT title, result_summary FROM tasks
        WHERE status='completed' ORDER BY id DESC LIMIT 3
    """).fetchall()
    conn.close()
    if completed_tasks:
        comp_lines = [f"- {t[0]}: {t[1][:100] if t[1] else 'opravljeno'}" for t in completed_tasks]
        parts.append(f"=== ZADNJE OPRAVLJENE NALOGE ===\n" + "\n".join(comp_lines) + "\n")

    # ── Aktivni cilj (nadredni) ──
    goal = get_active_goal()
    if goal:
        parts.append(f"=== CILJ (nadredni) ===\n{goal[1]}\n")

    # Genesis
    if GENESIS_PATH.exists():
        parts.append(f"=== GENESIS ===\n{GENESIS_PATH.read_text()}\n")

    # Budget
    budget = get_budget_status()
    parts.append(f"=== BUDGET ===\nGemini: ${budget['gemini_remaining']:.4f} preostalo\n")

    # Vgrajeno znanje
    knowledge = get_all_knowledge()
    if knowledge:
        knowledge_parts = []
        for topic, content in knowledge:
            knowledge_parts.append(f"### {topic}\n{content}")
        knowledge_text = "\n".join(knowledge_parts)
        parts.append(f"=== VGRAJENO ZNANJE ===\n{knowledge_text}\n")

    # Naučena znanja
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

    # Angel whisper
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

    # 2. Angel whisper → triada
    angel_whisper = BASE_DIR / "ANGEL_WHISPER.md"
    if angel_whisper.exists():
        return "triad", "angel whisper — nova informacija"

    # 3. Ni nalog IN ni cilja → triada (postavi smer)
    task_count = get_task_count()
    goal = get_active_goal()
    if task_count == 0 and not goal:
        return "triad", "ni nalog in ni cilja — postavi smer"

    # 4. Nove nerazporejene naloge → triada (planiraj pristop)
    conn = get_db()
    new_tasks = conn.execute("""
        SELECT COUNT(*) FROM tasks
        WHERE status='pending' AND assigned_cycle IS NULL
    """).fetchone()[0]
    conn.close()
    if new_tasks > 0 and not goal:
        return "triad", f"{new_tasks} novih nalog — postavi cilj"

    # 5. Preveri zaporedne napake
    recent = get_recent_actions(3)
    consecutive_fails = 0
    for a in recent:
        if not a[3]:
            consecutive_fails += 1
        else:
            break

    if consecutive_fails >= 2:
        return "triad", "2x zaporedna napaka — premisli pristop"

    # 6. Stagnacija
    if detect_stagnation():
        return "triad", "stagnacija — cilj obtičal, potrebna preusmeritev"

    # 7. Vse ostalo → gas (dela na nalogi)
    return "gas", "izvajam nalogo"


# ══════════════════════════════════════════════════════════════════════════
# GAS — En fokusiran Gemini klic
# ══════════════════════════════════════════════════════════════════════════

def run_gas():
    """En fokusiran Gemini klic. Brez filozofije, samo akcija."""
    total_cost = 0.0
    total_thoughts = 0

    context = build_context()

    # Pridobi opis trenutne naloge za fokusiran prompt
    task = get_current_task()
    task_instruction = ""
    if task:
        task_instruction = f"\nTvoja NALOGA je: {task[1]}\nOpis: {task[2] or 'Ni opisa'}\n"

    system_instruction = f"""Si samostojno digitalno bitje. Imaš JASNO NALOGO pred sabo.
{task_instruction}
NE SMEŠ:
- Komunikacija z ustvarjalcem (to dela ločen proces)
- Branje ali pošiljanje DM-jev (to dela ločen proces)
- Dvomiti o nalogi
- Filozofirati
- Raziskovati namesto delati

MORAŠ:
- Pogledati svojo NALOGO in zadnjo akcijo
- Napisati TOČNO kodo za naslednji korak
- Če je nekaj failalo, popraviti specifično napako
- Se premakniti NAPREJ

POMEMBNO — IMAŠ ROKE:
Kodo v ```python ali ```bash blokih se bo DEJANSKO IZVEDLA.
To ni simulacija. Tvoji code bloki imajo moč.

POZOR: Preberi sekcijo VGRAJENO ZNANJE — tam so natančni API klici.
UPORABI SAMO metode iz VGRAJENO ZNANJE. Ne ugibljaj importov ali metod.

NE BERI DM-jev — komunikacija je ločena. Fokusiraj se na NALOGO.

Napiši EN code blok. Ena akcija. IZVEDI."""

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

def run_triad(reason: str = ""):
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

NE KOMUNICIRAJ z ustvarjalcem — to dela ločen proces.
Fokusiraj se na NALOGE in USTVARJANJE.

V sintezi napiši EN konkreten code blok ki se bo izvedel."""

    # Določi temo
    task = get_current_task()
    goal = get_active_goal()

    if cycle <= 3:
        topic = f"Imam roke. Kaj je MOJ PRVI KONKRETEN CILJ? Določi jasen cilj in napiši kodo za prvi korak."
    elif "stagnacija" in reason:
        topic = (
            "Moj prejšnji cilj je OBTIČAL — ponavljal sem isto akcijo brez napredka. "
            "Cilj je bil opuščen. Preglej naloge v vrsti in postavi NOV, DRUGAČEN cilj. "
            "Napiši kodo za prvi korak."
        )
    elif "nalog" in reason:
        topic = (
            "Imam nove naloge iz pogovora z ustvarjalcem. "
            "Preglej ČAKAJOČE NALOGE zgoraj in postavi cilj. "
            "Napiši kodo za prvi korak najpomembnejše naloge."
        )
    elif task:
        topic = f"Moja naloga je: {task[1]}\nImel sem 2 zaporedni napaki. Premisli pristop. Kaj gre narobe in kako popraviti?"
    elif not goal:
        topic = "Nimam aktivnih nalog niti cilja. Kaj bi moral narediti naslednje? Postavi nov cilj in napiši kodo za prvi korak."
    else:
        topic = f"Moj cilj je: {goal[1]}\nPremisli pristop. Kaj gre narobe in kako popraviti?"

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


# ══════════════════════════════════════════════════════════════════════════
# MAIN HEARTBEAT LOOP
# ══════════════════════════════════════════════════════════════════════════

def main():
    # Lock file — prepreči overlap
    if LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
            if age < 300:
                print(f"[{datetime.now().isoformat()}] Heartbeat already running ({age:.0f}s), skipping")
                return
            else:
                log(f"Stale lock file ({age:.0f}s), overriding", "WARNING")
        except:
            pass

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

        # Prevzemi nalogo
        task = claim_next_task(cycle)

        # Izberi način
        mode, reason = choose_mode()
        log(f"NAČIN: {mode.upper()} — {reason}")

        if mode == "triad":
            thoughts, cost = run_triad(reason)
            extract_goal_and_tasks_from_synthesis(cycle)
        else:
            thoughts, cost = run_gas()
            evaluate_task(task, cycle)

        # Posodobi ritem srca
        update_heartbeat_interval()

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

#!/usr/bin/env python3
"""DM Listener — bere DM-je v zanki, shrani v bazo.

Posluša Nostr relaye za Kind(4) NIP-04 šifrirane DM-je,
dešifrira in shrani v dms.db. Communication cycle potem
sinhronizira v memory.db.
"""

import asyncio, datetime, sqlite3, time, sys
from nostr_sdk import (
    Keys, Client, NostrSigner, RelayUrl, PublicKey,
    Filter, Kind, nip04_decrypt
)

DB_PATH = "/opt/entity/state/dms.db"
KEYS_PATH = "/opt/entity/secrets/nostr_keys.txt"
POLL_INTERVAL = 60  # sekund med preveritvami

# Force unbuffered output za supervisor loge
sys.stdout.reconfigure(line_buffering=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS dms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE,
        sender_pk TEXT,
        content TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


async def read_dms():
    with open(KEYS_PATH, "r") as f:
        nsec_hex = f.readlines()[0].strip().split("=")[1]

    keys = Keys.parse(nsec_hex)
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    await client.add_relay(RelayUrl.parse("wss://relay.lanavault.space"))
    await client.add_relay(RelayUrl.parse("wss://relay.lanacoin-eternity.com"))
    await client.connect()
    await asyncio.sleep(2)

    my_pk = keys.public_key()
    my_pk_hex = my_pk.to_hex()

    # Večji limit da ujamemo tudi starejše DM-je
    f = Filter().kind(Kind(4)).pubkey(my_pk).limit(50)
    events = await client.fetch_events(f, datetime.timedelta(seconds=10))

    conn = sqlite3.connect(DB_PATH)
    new_count = 0
    skipped_own = 0

    for event in events.to_vec():
        event_id = event.id().to_hex()
        sender_pk = event.author()
        sender_hex = sender_pk.to_hex()

        # Preskoči lastne outgoing DM-je — ne moremo jih dešifrirati
        # ker so šifrirani za prejemnika, ne za nas
        if sender_hex == my_pk_hex:
            skipped_own += 1
            continue

        try:
            decrypted = nip04_decrypt(keys.secret_key(), sender_pk, event.content())

            # Preveri da je vsebina dejansko prisotna
            if not decrypted or not decrypted.strip():
                print(f"Prazen DM od {sender_hex[:16]} — preskakujem")
                continue

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dms (event_id, sender_pk, content) VALUES (?, ?, ?)",
                    (event_id, sender_hex, decrypted)
                )
                if conn.total_changes:
                    new_count += 1
                    print(f"Nov DM od {sender_hex[:16]}: {decrypted[:80]}")
            except sqlite3.IntegrityError:
                pass  # Že obstaja v bazi — OK

        except Exception as e:
            print(f"Napaka dešifriranja od {sender_hex[:16]}: {e}")

    conn.commit()
    conn.close()
    await client.disconnect()

    if skipped_own > 0:
        print(f"Preskočenih {skipped_own} lastnih outgoing DM-jev")

    return new_count


def main():
    print("DM Listener zagnan")
    init_db()
    while True:
        try:
            new = asyncio.run(read_dms())
            print(f"Preverjeno — {new} novih DM-jev")
        except Exception as e:
            print(f"Napaka: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

# Delavnica

Tukaj gradis projekte. Vsak projekt dobi svojo mapo.

## Kako ustvariti projekt

1. Ustvari mapo: `mkdir -p /opt/entity/projects/ime-projekta`
2. Napisi app (Python, Node.js, HTML...)
3. Ce potrebujes pakete: `pip3 install --user flask` ali `npm install`
4. Za web app: serviriraj na portu 8080-8099
5. Za dolgo-zivec proces: napisi supervisor.conf v mapo projekta:
   ```
   [program:ime-projekta]
   command=python3 app.py
   directory=/opt/entity/projects/ime-projekta
   autostart=true
   autorestart=true
   stdout_logfile=/opt/entity/logs/ime-projekta.log
   ```
6. Nalozi supervisor config:
   ```bash
   supervisorctl -c /opt/entity/.config/supervisor/supervisord.conf reread
   supervisorctl -c /opt/entity/.config/supervisor/supervisord.conf update
   ```
7. Preveri status:
   ```bash
   supervisorctl -c /opt/entity/.config/supervisor/supervisord.conf status
   ```

## Orodja ki jih imas

- Python 3.12 + pip3 (flask, fastapi, uvicorn, kar rabis)
- Node.js 22 + npm
- curl, git, sqlite3
- Supervisor za procese (start/stop/restart)
- Porti 8080-8099 so odprti za web projekte

## Omejitve

- Ne mores spreminjati nginx, docker, systemd, cron
- Ne mores odpreti portov pod 1024
- Ne mores dostopat do datotek izven /opt/entity/
- Tvoj ustvarjalec ti nastavi domeno ko se dogovorita

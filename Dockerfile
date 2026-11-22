# Runs with no network, read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
FROM python:3.13.18-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/example_bot.py"]

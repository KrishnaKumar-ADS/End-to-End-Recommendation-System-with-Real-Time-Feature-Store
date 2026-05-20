# Demo Script (5 minutes)

## Pre-demo setup

1. Start stack:

```bash
make dev
```

2. Confirm services:

```bash
docker compose -f docker/docker-compose.yml ps
```

3. Open tabs:

- http://localhost:3000
- http://localhost:8000/docs

## Scene timeline

### Scene 1 (0:00 - 0:30)

- Show one-command startup.
- Show service health.

Narration:

"The stack starts with one command and brings up frontend, API, worker, and data services."

### Scene 2 (0:30 - 1:30)

- Submit a repository ingestion request from UI.
- Show status progression.

### Scene 3 (1:30 - 2:30)

- Ask a repository question in chat.
- Highlight streaming tokens and source citations.

### Scene 4 (2:30 - 3:30)

- Open graph page.
- Click node and show details panel.

### Scene 5 (3:30 - 4:30)

- Run semantic search in UI.
- Show ranked results.

### Scene 6 (4:30 - 5:00)

- Show benchmark and provider comparison reports.
- Close with deployment docs pointer.

## Recording tips

- Use 1920x1080 resolution.
- Keep terminal font readable (at least 16px equivalent).
- Avoid background notifications.
- Keep narration focused on outcomes and user value.

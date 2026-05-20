# Deployment Guide

This document covers practical deployment options for the AI Codebase Intelligence platform.

## Prerequisites

- Docker Engine 24+
- Docker Compose plugin 2+
- A Linux host with at least 2 vCPU, 4 GB RAM
- API keys configured in .env.prod

Required environment variables:

- GEMINI_API_KEY
- OPENROUTER_API_KEY
- POSTGRES_PASSWORD
- NEO4J_PASSWORD

## Option A: DigitalOcean Droplet

Recommended size:

- Basic Droplet
- 2 vCPU / 4 GB RAM
- Ubuntu 22.04

Estimated cost: around $12/month.

### 1) Provision host

```bash
ssh root@YOUR_DROPLET_IP
apt update
apt install -y docker.io docker-compose-plugin git ufw
systemctl enable docker
systemctl start docker
```

### 2) Clone and configure

```bash
git clone https://github.com/your-username/ai-codebase-intelligence
cd ai-codebase-intelligence
cp .env.prod.example .env.prod
# edit .env.prod and set secrets
```

### 3) TLS and stack startup

```bash
make gen-certs
make prod-up
make prod-ps
```

### 4) Firewall

```bash
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
```

### 5) Verify

```bash
curl -k https://localhost/health
```

Expected: HTTP 200 and JSON status.

## Option B: Hetzner Cloud

Recommended instance: CX21 (2 vCPU, 4 GB RAM).

Estimated cost: around EUR 5-6/month.

Setup steps are the same as DigitalOcean after SSH access:

1. Install Docker and Compose plugin
2. Clone repository
3. Configure .env.prod
4. Run make gen-certs and make prod-up
5. Open ports 80 and 443 in Hetzner firewall rules

## Option C: Render

Render can host parts of the stack, but full local graph/vector services are harder on free plans.

Suggested split:

- Backend: Render Web Service (Docker)
- Frontend: Render Static Site or Web Service
- Neo4j: external hosted instance (for example Neo4j Aura)
- Qdrant: managed instance or self-hosted VM

Notes:

- Set environment variables in Render dashboard.
- Ensure backend can reach external Neo4j and Qdrant.
- Keep APP_ENV=production.

## Restart on host reboot (systemd)

Create /etc/systemd/system/codebase-intelligence.service:

```ini
[Unit]
Description=AI Codebase Intelligence Stack
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/root/ai-codebase-intelligence
ExecStart=/usr/bin/docker compose -f docker/docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker/docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
systemctl daemon-reload
systemctl enable codebase-intelligence
systemctl start codebase-intelligence
```

## Post-deploy checklist

- /health returns 200
- /docs loads over HTTPS
- frontend root page loads
- POST /api/v1/ingest accepts requests
- smoke test script passes

Run smoke test:

```bash
BASE_URL=https://YOUR_DOMAIN SKIP_TLS=0 bash scripts/smoke-test.sh
```

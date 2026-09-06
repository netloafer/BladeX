# Security Policy

## Supported versions

BladeX is in beta. Security fixes target the latest release line.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, email the maintainers privately (see the repo's `SECURITY.md` contact or GitHub's "Report a vulnerability" flow on the Advisories tab). Include:

- A description of the issue and its impact.
- Steps to reproduce (proof of concept, logs, or a minimal request).
- Your assessment of who could be affected.

We'll acknowledge within a reasonable window and coordinate a fix + disclosure timeline. Responsible disclosure is appreciated; we won't pursue good-faith research.

## Threat model & operational security

BladeX is a self-hosted proxy that **stores your conversation memory locally and forwards requests to an upstream LLM**. The security posture you need depends entirely on your network exposure.

### Binding (the most important knob)

- **Default bind is `127.0.0.1`** - only your local machine can reach the proxy. This is safe for solo self-host.
- If you set `BLADEX_HOST` to a non-loopback address (e.g. `0.0.0.0`) **and** leave `BLADEX_AUTH_ENABLED=false` (the default), BladeX prints a **`BLADEX SECURITY WARNING`** banner at startup and logs `bind_non_loopback_no_auth`. Do not ignore it.

### Endpoint exposure

| Endpoint | Auth | Risk if exposed |
|---|---|---|
| `/v1/*` | client key when `AUTH_ENABLED=true`; **open when false** | anyone can call your upstream LLM (billed to you) and read your injected memory |
| `/admin/*` | `BLADEX_ADMIN_KEYS` when set, otherwise falls back to the client key; **open when `AUTH_ENABLED=false`** | anyone can delete turns (tombstone), rewrite/merge Matters, promote scope - i.e. tamper with your memory |
| `/metrics` | none | operational data: request counts, latency, routing distribution, queue depth |
| `/health`, `/ready` | none | liveness / readiness booleans, P1 overflow counts |

**Separate the management plane.** Set `BLADEX_ADMIN_KEYS` (same `key||label` format as
`BLADEX_CLIENT_KEYS`) so that `/admin/*` requires its own credential. Without it, the key you
hand to an agent also carries full destructive admin rights over your memory. `bladex init`
generates both keys for you.

- **Personal deployment, `BLADEX_ADMIN_KEYS` unset**: the admin plane falls back to the client
  key (unchanged behaviour) and the proxy logs `admin_key_shared_with_data_plane` at startup.
- **`config/identity.toml` present (multi-principal deployment)**: a separate admin key is
  **required** — the proxy refuses to start without one, because a shared key means any
  employee's agent key can administer the whole store.

With auth off entirely, **admin write operations are unprotected** — this is the most dangerous
exposure on a non-loopback bind.

### Hardening for network exposure

If anything other than your laptop can reach the proxy:

1. **Set `BLADEX_AUTH_ENABLED=true`** and generate strong `BLADEX_CLIENT_KEYS` (`python3 -c "import secrets; print('bladex-' + secrets.token_urlsafe(24))"`).
2. **Front with a TLS reverse proxy** (nginx/caddy) - BladeX itself speaks HTTP only. Examples in `docs/deployment-selfhost.md`.
3. **Restrict `/metrics` and `/health`** to an internal/monitoring network via ACL, or protect them at the reverse proxy with basic auth.
4. Keep `.env` (which holds keys) out of git - it's gitignored; verify your `BLADEX_CLIENT_KEYS` are strong and unique.

### Data flow

- Embedding (e5) runs **fully locally**; no text leaves for vectorization.
- The only outbound data is the request you send to your configured upstream LLM (plus the injected memory block). Choose your upstream accordingly.
- Memory lives in `data/` (RocksDB + LanceDB). Protect that directory; `bladex export` / sync connectors move it only when you configure them.

## Not in scope

BladeX does not provide transport-level guarantees against an attacker who has already compromised the host it runs on. At-rest encryption of the data directory is the operator's responsibility (e.g. FileVault/LUKS on the host).

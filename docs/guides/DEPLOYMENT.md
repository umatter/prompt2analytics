# Deploying the web app (bring-your-own-key)

This guide covers hosting the p2a web UI publicly — e.g. a demo for reviewers —
in **bring-your-own-key (BYOK)** mode, where each visitor supplies their own LLM
API key in the browser.

## Two pieces, hosted separately

Unlike a purely static site (e.g. the Prompt Arena classroom app, which is a
static frontend + a serverless Cloudflare Worker), p2a has a **heavyweight,
stateful backend** that cannot run on Workers/Pages:

| Piece | What it is | Where it runs |
|-------|-----------|---------------|
| **Frontend** | Dioxus → static WASM + assets | GitHub Pages (free, fits the qamelab pattern) |
| **Backend** | `p2a-mcp` axum server (270 tools, SurrealDB/RocksDB, ~575 MB image) | An always-on host with HTTPS (VPS / Fly.io / Render / a box) |

The frontend half mirrors how the other qamelab sites deploy. The backend is the
genuinely new piece — it needs a real server, so it gets its own HTTPS endpoint
(e.g. `https://api.p2a.qamelab.org`) that the frontend is built to talk to.

## Frontend: GitHub Pages on a qamelab subdomain

The repo lives under `umatter/`, not the `qamelab` org, so a path like
`qamelab.org/p2a` (how Prompt Arena is served) isn't available — a **subdomain**
CNAME'd to GitHub Pages is the clean route. The `.github/workflows/web-deploy.yml`
workflow builds the WASM and publishes it; it reads two repo **Variables**
(Settings → Secrets and variables → Actions → Variables):

- `P2A_BACKEND_URL` — public HTTPS URL of the backend, baked into the WASM at
  build time (e.g. `https://api.p2a.qamelab.org`).
- `WEB_DOMAIN` — the custom domain, written to `CNAME` (e.g. `p2a.qamelab.org`).

Deploy steps:

1. Set the two repo Variables above.
2. Add a DNS record at the qamelab.org zone:
   `CNAME p2a.qamelab.org → umatter.github.io.`
3. Deploy the backend (next section) and note its HTTPS URL → that's
   `P2A_BACKEND_URL`.
4. Run the **Deploy Web App** workflow (push to `main` or trigger manually).
5. Settings → Pages → Source: **GitHub Actions**; set the custom domain to
   `p2a.qamelab.org` and tick **Enforce HTTPS** once the cert provisions.
6. Add `https://p2a.qamelab.org` to the backend's `P2A_CORS_ORIGINS`.
7. Add the live URL to the qamelab site card (`qamelab.github.io`,
   `src/content/software/p2a.md` → `liveUrl: https://p2a.qamelab.org`).

## Threat model in one paragraph

As of **v0.1.2** the `p2a-mcp` backend supports two safe postures for a public
bind, and refuses to start on a non-loopback address that has neither:

- **Authenticated** — set `P2A_ACCESS_TOKEN`; every `/api` route then requires
  `Authorization: Bearer <token>` (`/health` and CORS preflight are exempt).
  Use this to restrict *who* can reach the demo.
- **Open but hardened** — set `P2A_ALLOW_UNAUTHENTICATED=1` for a no-login BYOK
  demo. There are no user accounts; per-visitor isolation relies on the
  **unguessable, non-enumerable session ID** the frontend mints (a capability,
  like an unlisted share link). In this mode the two enumeration endpoints
  (`GET /api/sessions` list and `GET /api/files`) are **not registered**, so
  nobody can list other sessions or browse the filesystem.

Either way, the v0.1.2 code hardening closes the server-exploitation vectors: an
allowlist on the LLM `base_url` blocks SSRF, the SQL tools reject file-reading
functions with literal paths (no `read_csv_auto('/etc/passwd')`), and the
data-file parsers bound-check declared sizes (no allocation-DoS). BYOK stays
safe **only because no shared LLM key is present** on the backend. The must-dos
below keep it that way, encrypt traffic, confine the filesystem, and push
abuse/DoS control to the edge (open mode has no per-request auth to throttle
behind).

## Must-do checklist

1. **Do NOT set LLM API keys on the backend.**
   Leave `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` **unset** in
   the backend's environment. If present, the backend falls back to them for any
   request that omits a key (`transport/http.rs`), turning an unauthenticated
   public endpoint into an open proxy billed to your account. Unset = true BYOK.

2. **Terminate TLS in front of the backend.**
   BYOK keys travel in request bodies; without HTTPS they're exposed on the wire.
   Run a reverse proxy (Caddy/nginx/Traefik) or a platform that provides HTTPS.

3. **Lock CORS to your frontend origin.**
   Set `P2A_CORS_ORIGINS=https://your-demo.example` (comma-separated for several).
   Never use `--cors-permissive` / `P2A_CORS_PERMISSIVE` in production.

4. **Rate-limit and time out at the edge.**
   Add per-IP rate limiting and request timeouts at a reverse proxy or CDN — the
   *compute* (regressions, ML) runs on your server even though LLM cost is the
   user's. This is **the** abuse/DoS control in open-unauthenticated mode (there
   is no per-request auth to throttle behind), so do not skip it. A CDN in front
   (e.g. **Cloudflare**, free tier: a Rate Limiting rule on `/api/*`, plus WAF
   for obvious abuse) works well and is invisible to reviewers; a reverse proxy
   (Caddy/nginx) rule works too. **Exclude the SSE path**
   `/api/llm/chat/stream` from response buffering and idle timeouts, or
   streaming chat will break.

5. **Cap request body size.**
   `P2A_MAX_HTTP_BODY_MB` (default 32) limits memory from oversized uploads.

6. **Confine the filesystem jail (`P2A_DATA_ROOT`).**
   Set `P2A_DATA_ROOT` to a dedicated, **empty** directory (e.g. `/data`, which
   the backend image creates) so any filesystem/DB tool can only reach that
   directory. Unset, it defaults to the process working directory — never leave
   it defaulting to a home directory on an exposed host. The directory must
   exist (it is canonicalized at startup).

7. **Run the container locked down.**
   `docker-compose.yml` ships with `cap_drop: ALL`, `no-new-privileges`,
   `pids_limit`, and memory/CPU limits. Writable state is confined to the
   `p2a-data` volume and a `/tmp` tmpfs; enable `read_only: true` after verifying
   those are the only write paths.

## API-key handling (for transparency)

User keys are **never persisted or logged** server-side. Each request constructs
a per-request provider, uses the key only to set the outbound `Authorization`
header to OpenAI/Anthropic, then drops it. There is no server-side key store
(the former unused `settings.api_key_encrypted` field was removed). The keys'
only exposure point is **in transit** — hence the TLS requirement above. Note
the frontend's "stored locally, never sent to our servers" copy is accurate for
desktop/self-host but, in a hosted setup, the key does transit your backend to
reach the LLM.

## Example: Caddy reverse proxy

```caddy
your-demo.example {
    encode gzip
    # Per-IP rate limit (requires the caddy-ratelimit plugin), excluding SSE.
    @stream path /api/llm/chat/stream
    reverse_proxy @stream backend:8080 {
        flush_interval -1   # disable buffering for SSE
    }
    reverse_proxy backend:8080
}
```

## What this does NOT give you

The open-but-hardened mode is appropriate for an **open demo with disposable,
isolated sessions**. It is *not* a multi-tenant SaaS: there are no user
accounts and no server-side key storage, and per-visitor isolation is only as
strong as keeping session IDs secret (they are unguessable and non-enumerable,
but treat a leaked session URL like a leaked share link). Anonymous callers can
still *use compute* (bounded by your edge rate limit and machine size) — that is
the accepted trade for a zero-friction demo. If you need to restrict *who* can
use it, switch to the authenticated posture (`P2A_ACCESS_TOKEN`) and distribute
the token; if you need real multi-tenant accounts and server-side key storage,
add an identity layer in front of the backend (KMS-managed master key — see the
discussion in project memory).

## Fly.io quick reference

`deploy/fly.toml` is configured for the open-but-hardened posture: it pins an
explicit image tag, sets `P2A_ALLOW_UNAUTHENTICATED=1`, `P2A_DATA_ROOT=/data`,
and locks CORS. Ship a new backend build with:

```bash
# 1. Build & publish the image for the release tag (from main):
#    push tag vX.Y.Z, then run the "Backend Image" workflow via workflow_dispatch
#    on main so the vX.Y.Z tag is published to GHCR.
# 2. Point deploy/fly.toml [build].image at ghcr.io/.../backend:X.Y.Z
# 3. Deploy:
fly deploy -c deploy/fly.toml
```

Put Cloudflare (or your CDN) in front of `api.p2a.qamelab.org` with a Rate
Limiting rule on `/api/*` for abuse/DoS control.

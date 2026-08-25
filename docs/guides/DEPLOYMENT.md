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
(e.g. `https://p2a-api.qamelab.org`) that the frontend is built to talk to.

## Frontend: GitHub Pages on a qamelab subdomain

The repo lives under `umatter/`, not the `qamelab` org, so a path like
`qamelab.org/p2a` (how Prompt Arena is served) isn't available — a **subdomain**
CNAME'd to GitHub Pages is the clean route. The `.github/workflows/web-deploy.yml`
workflow builds the WASM and publishes it; it reads two repo **Variables**
(Settings → Secrets and variables → Actions → Variables):

- `P2A_BACKEND_URL` — public HTTPS URL of the backend, baked into the WASM at
  build time (e.g. `https://p2a-api.qamelab.org`).
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

2. **Pin the LLM egress allowlist (`P2A_LLM_ALLOWED_HOSTS`).**
   Callers supply their own provider `base_url`, so set
   `P2A_LLM_ALLOWED_HOSTS=api.openai.com,api.anthropic.com,openrouter.ai`
   (bare hostnames, matched exactly and case-insensitively against the URL
   host). Leave it unset and the LLM routes will POST a caller-chosen body to
   any https host and relay the response back — an SSRF and an open relay
   through your egress IP, carrying whatever key the caller supplied. The
   scheme and private-IP checks still apply when it is unset, but every public
   host is reachable. Setting it also revokes the Ollama loopback exemption,
   which a remote caller would otherwise use to read the backend's own ports; a
   hosted backend cannot reach a user's local Ollama regardless.

3. **Terminate TLS in front of the backend.**
   BYOK keys travel in request bodies; without HTTPS they're exposed on the wire.
   Run a reverse proxy (Caddy/nginx/Traefik) or a platform that provides HTTPS.

4. **Lock CORS to your frontend origin.**
   Set `P2A_CORS_ORIGINS=https://your-demo.example` (comma-separated for several).
   Never use `--cors-permissive` / `P2A_CORS_PERMISSIVE` in production.

5. **Rate-limit and time out at the edge.**
   Add per-IP rate limiting and request timeouts at a reverse proxy or CDN — the
   *compute* (regressions, ML) runs on your server even though LLM cost is the
   user's. This is **the** abuse/DoS control in open-unauthenticated mode (there
   is no per-request auth to throttle behind), so do not skip it. A CDN in front
   (e.g. **Cloudflare**, free tier: a Rate Limiting rule on `/api/*`, plus WAF
   for obvious abuse) works well and is invisible to reviewers; a reverse proxy
   (Caddy/nginx) rule works too. **Exclude the SSE path**
   `/api/llm/chat/stream` from response buffering and idle timeouts, or
   streaming chat will break.

   Putting Cloudflare in front does not break the SSE path, as long as you do
   not disable the server's keep-alive: the stream emits an axum keep-alive
   every 15s against a 900s Proxy Idle Timeout, and the 125s Proxy Read Timeout
   applies to the response *headers*, which the handler sends immediately. A
   long tool loop therefore does not trip a 524. If you ever remove the
   keep-alive, that stops being true.

   `[http_service.concurrency]` in `deploy/fly.toml` caps in-flight requests,
   but it is a backstop, not a substitute: it cannot distinguish one abusive
   caller from several legitimate ones.

6. **Cap request body size.**
   `P2A_MAX_HTTP_BODY_MB` (default 32) limits memory from oversized uploads.

7. **Confine the filesystem jail (`P2A_DATA_ROOT`).**
   Set `P2A_DATA_ROOT` to a dedicated, **empty** directory (e.g. `/data`, which
   the backend image creates) so any filesystem/DB tool can only reach that
   directory. Unset, it defaults to the process working directory — never leave
   it defaulting to a home directory on an exposed host. The directory must
   exist (it is canonicalized at startup).

8. **Run the container locked down.**
   `docker-compose.yml` ships with `cap_drop: ALL`, `no-new-privileges`,
   `pids_limit`, and memory/CPU limits. Writable state is confined to the
   `p2a-data` volume and a `/tmp` tmpfs; enable `read_only: true` after verifying
   those are the only write paths.

## API-key handling (for transparency)

User keys are **never persisted or logged** server-side. Each request constructs
a per-request provider, sets one outbound auth header — `Authorization: Bearer`
for OpenAI-compatible providers including OpenRouter, `x-api-key` for Anthropic
— then drops the key. There is no server-side key store (the former unused
`settings.api_key_encrypted` field was removed). The keys' only exposure point
is **in transit** — hence the TLS requirement above.

**Transit is web-only.** Desktop and mobile builds start an embedded backend on
loopback, so the key never leaves the device there. On a hosted deployment it
does pass through your backend, on every route that carries a provider config:
`/api/llm/chat`, `/api/llm/chat/stream`, and `/api/llm/generate-title`. The
frontend says so — "sent with each request through our backend to your chosen
provider" — so keep this section and that copy in agreement.

**The destination is only as constrained as `P2A_LLM_ALLOWED_HOSTS`.** Callers
supply their own `base_url`, so on an exposed deployment this variable is not
optional: without it the LLM routes will POST to any https host a caller names
and relay the response back, which is an SSRF and an open relay through your
egress IP, using whatever key the caller supplied. See step 2 above.

**Keys are the smaller half of the story.** Every tool call also sends the
user's dataset, prompt, and tool arguments to your server — that is the larger
disclosure, and it is inherent to running the analytics server-side. Tool
arguments are not written to disk unless you set `P2A_AUDIT_LOG`; leave it
unset on a demo, or say plainly that you have enabled it.

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

Put Cloudflare (or your CDN) in front of `p2a-api.qamelab.org` with a Rate
Limiting rule on `/api/*` for abuse/DoS control.

**Keep the backend hostname one level deep.** It is `p2a-api.qamelab.org` and not
`api.p2a.qamelab.org` for a reason: on a Cloudflare full setup, Universal SSL
covers the apex and *first-level* subdomains only, and a `*.example.com`
wildcard does not match `api.staging.example.com`. Proxying a second-level host
therefore leaves the edge with no certificate to present, and it aborts the TLS
handshake outright — the origin stays healthy while every client gets a
handshake failure, and `curl -k` fails too, since nothing is served to
distrust. Covering a deeper name needs Advanced Certificate Manager, Total TLS,
or an uploaded wildcard. Renaming is free; those are not.

Fly issues its own certificate for the origin, so add the hostname
DNS-only first (`fly certs add`, then A/AAAA records with the proxy off) and
only enable the proxy once `fly certs check` reports `Issued`. Verify TLS on the
new hostname *before* repointing `P2A_BACKEND_URL` at it.

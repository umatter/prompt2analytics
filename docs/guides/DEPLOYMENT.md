# Deploying the web app (bring-your-own-key)

This guide covers hosting the p2a web UI publicly — e.g. a demo for reviewers —
in **bring-your-own-key (BYOK)** mode, where each visitor supplies their own LLM
API key in the browser.

## Threat model in one paragraph

The `p2a-mcp` HTTP backend has **no enforced authentication** (the
`P2A_AUTH_ENABLED` / `P2A_JWT_SECRET` config exists but is *not* wired into the
router) and exposes all analytics tools, including data-loading and database
tools. Anyone who can reach the port can call them. BYOK is safe **only because
no shared LLM key is present** on the backend — so the must-dos below are about
(a) keeping it that way, (b) encrypting traffic, and (c) limiting blast radius.

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

4. **Rate-limit and time out at the proxy.**
   Add per-IP rate limiting and request timeouts at the reverse proxy — the
   *compute* (regressions, ML) runs on your server even though LLM cost is the
   user's. **Exclude the SSE path** `/api/llm/chat/stream` from response
   buffering and idle timeouts, or streaming chat will break.

5. **Cap request body size.**
   `P2A_MAX_HTTP_BODY_MB` (default 32) limits memory from oversized uploads.

6. **Run the container locked down.**
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

This setup is appropriate for an **open demo with disposable, isolated sessions**.
It is *not* a multi-tenant SaaS: there are no user accounts, no per-user
authorization, and no server-side key storage. If you need those, add real
authentication in front of the backend (and only then consider server-side key
storage, with a KMS-managed master key — see the discussion in project memory).

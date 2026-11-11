# Security policy

## Reporting
This is a portfolio project without a production deployment. Report vulnerabilities privately through GitHub
security advisories on this repository. Please do not open public issues for vulnerabilities.

## Current security posture (M0–M4)

### Inference server (M4)
Threat model: untrusted clients on the network; the operator and the model checkpoint are trusted. Out of scope: TLS
(terminated by the router/ingress, M8/M9), multi-tenant isolation beyond per-key identity and scheduler fairness.

| Threat | Control |
|---|---|
| Unauthenticated use | Auth on by default: `Authorization: Bearer dli_<id>_<secret>`. The server refuses to start without a key file, and refuses `--no-auth` on a non-loopback address. Binds to 127.0.0.1 by default |
| Key theft from the server | Only Argon2id hashes are stored (`python -m dli.server.auth new <tenant>` prints the key once) |
| Key enumeration / timing | Lookup by key id with a dummy Argon2 verification for unknown ids; verification runs off the event loop |
| Pre-auth resource use | Auth and the body-size limit (413) run as ASGI middleware **before** the body is parsed |
| Tenant spoofing | Tenant comes from the authenticated key; a client `X-Tenant-Id` header is ignored |
| Resource exhaustion | In-flight and queue caps (429 + `Retry-After`), deadline admission, `max_tokens` / context / stop-string limits, request timeout, KV capacity checks, client-disconnect abort |
| Template injection | Chat templates render in Jinja's immutable sandbox; template errors are 400s, not crashes |
| Information exposure | Metrics and drain controls are on a separate loopback listener with an optional bearer token (constant-time compare); error bodies carry no stack traces |

Unsupported OpenAI parameters are rejected rather than ignored, so clients never get silently different behaviour.

### Supply chain and development
- Model weights are loaded **only** from `safetensors` files (no pickle deserialization) at pinned Hugging Face revisions,
  with strict tensor name and shape validation.
- Dependencies are fully pinned in `uv.lock` and audited with `pip-audit` for both PyTorch builds. Security floors for
  transitive dependencies are declared in `pyproject.toml` (`constraint-dependencies`).
- Secrets are never committed (gitleaks pre-commit hook and CI scan).
- GPU development host: no credentials are stored there, SSH agent forwarding is disabled, and services bind to localhost
  only (`docs/operations.md`).

## Disclosure log
| Date | Item | Severity | Resolution |
|---|---|---|---|
| 2026-11-10 | Commit `207083c` published profiler output with absolute paths revealing the developer's local username and the repo's directory layout | Low: no credentials, keys, infrastructure addresses, or secrets (verified by pattern scan and gitleaks on the commit) | Redacted going forward; profile output is now path-redacted by the benchmark script; every push greps for personal paths. Published history intentionally left intact (no force-push) |

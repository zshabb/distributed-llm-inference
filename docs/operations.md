# Operations

This runbook covers the **inference server** (M4, section 0), the **Vultr A16 GPU host** (sections 1-5) and
**provider-independent GPU host profiles, including the RunPod A5000** (section 6).
Router, deployment, alerting, and incident runbooks are added as those components land (M8, M9).

## 0. Running the inference server

```bash
# 1. Create API keys. The key is printed once (stderr); only its Argon2id hash goes in the file.
uv run --frozen python -m dli.server.auth new acme            # prints {"key_id": ..., "secret_hash": ...}
#    Put the printed JSON objects in a list: [ {...}, {...} ] -> keys.json (keep it out of Git)

# 2. Serve (defaults: API on 127.0.0.1:8000, admin on 127.0.0.1:8001, auth required)
uv run --frozen dli-serve --model HuggingFaceTB/SmolLM2-135M-Instruct --api-keys-file keys.json --kv-cache-gb 1

# Local development only: no auth, refused unless --host is a loopback address
uv run --frozen dli-serve --model HuggingFaceTB/SmolLM2-135M-Instruct --no-auth
```

Any OpenAI client works: `OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<key>")`. `--model` takes a local
checkpoint directory or a repo id from the pinned list in `dli.model.hub` (unpinned ids are refused).

| Concern | Behaviour | Flags |
|---|---|---|
| Binding | Loopback by default. A public `--host` requires API keys | `--host`, `--port` |
| Authentication | `Authorization: Bearer dli_<id>_<secret>`; Argon2id hashes verified off the event loop, before the body is parsed. Tenant comes from the key; client `X-Tenant-Id` is ignored | `--api-keys-file`, `--no-auth` |
| Admin listener | `/metrics` (Prometheus), `/admin/state`, `/admin/drain`, `/admin/undrain` on a separate loopback port. Optional bearer token, compared in constant time | `--admin-host`, `--admin-port`, `--admin-token-env` (default `DLI_ADMIN_TOKEN`) |
| Overload | 429 + `Retry-After` when in-flight or queued limits are reached, or an `X-Deadline-Ms` budget cannot be met (predicted from the measured prefill rate); a request whose deadline passes while queued is aborted with 429 before any byte is sent | `--max-inflight-requests`, `--max-queued-requests` |
| Limits | Body ≤ 2 MiB (413), `max_tokens` ≤ 4096 and ≤ context, ≤ 4 stop strings, non-streaming timeout 504 | `--request-timeout` |
| KV pool | Fixed at startup | `--num-blocks` or `--kv-cache-gb`, `--block-size` |
| Shutdown | SIGINT/SIGTERM: new requests get 503 and `/readyz` fails, in-flight requests finish (bounded), then exit 0 | `--drain-timeout` |
| Health | `/healthz` (process up), `/readyz` (engine healthy and not draining) on the API port | |

Clients that disconnect mid-stream are detected and their engine request is aborted, which frees its KV blocks. Engine
failure marks the server unhealthy: new requests get 503 and streaming requests receive an in-band error event.

## 1. GPU host model

| Property | Value |
|---|---|
| Role | Single-GPU (tier T2) execution and measurement over SSH. Never holds source of truth, Git credentials, or secrets |
| Hardware (verified 2026-11-10) | Vultr cloud GPU: 1× NVIDIA A16 (GA107, compute capability 8.6, 10 SMs, 15,356 MiB / 14.61 GiB usable, ECC on, **PCI pass-through**, not vGPU, so no license throttling), 6 vCPU Xeon (Cascade Lake, Hyper-V guest), 58 GiB RAM, 328 GB disk |
| Software baseline (verified) | Ubuntu 22.04.5, kernel 5.15.0-191, driver 580.95.05 (CUDA 13.0), gcc present (Triton JIT works), CUDA toolkit 13.4 under `/usr/local/cuda` (unused: wheels bundle their runtime). Preinstalled by the image: Docker 29.8.1, NVIDIA Container Toolkit 1.20.0 with the `nvidia` runtime, DCGM host engine on 127.0.0.1:5555. Evidence: `benchmarks/results/2026-11-10-GPU-A16x1-vultr-env-validation/` |
| Python env | `uv` 0.13.5 (checksum-verified) → Python 3.12 → `uv sync --frozen --extra cu130` (torch 2.14.1+cu130 from `uv.lock`) |
| Evidence label | `[GPU:A16x1@vultr]`. Single-GPU only: never multi-GPU or multi-node evidence (ENGINEERING_STANDARDS §3) |

### Trust boundary
```
 laptop (WSL2): repo + .git, ssh-agent holding the key ──ssh (no agent forwarding)──► host: ~/work/distributed-llm-inference
                ◄────────────── rsync results (benchmarks/results/) ──────────────────
```
- The private key stays on the laptop. Scripts pass only the **public** key to `ssh -i` with `IdentitiesOnly=yes`,
  so the agent signs. `ForwardAgent=no` is set explicitly.
- The host receives source files only (`git ls-files`), plus a `REVISION` file naming the exact commit and dirty state.
- Services started on the host bind to `127.0.0.1`. Use `ssh -L <port>:127.0.0.1:<port>` for access. Docker publishes
  ports on `127.0.0.1` by default (`/etc/docker/daemon.json` `"ip"`), and the Docker API is unix-socket only.
  These are **defense-in-depth layers, not the network security boundary**: a container started with an explicit
  `-p 0.0.0.0:…` still binds publicly, and `--network host` bypasses the default entirely. The primary controls remain the
  owner-managed Vultr cloud firewall and explicit, authenticated service configuration.
- SSH daemon settings, host firewall, and the Vultr cloud firewall are owner-managed and never changed by these scripts.
  `inspect_host.sh` reads them for the record.

## 2. One-time local setup

```bash
cp scripts/gpu/host.env.example scripts/gpu/host.env      # untracked; set GPU_HOST=root@<ip>
# Load the key into the local agent yourself (scripts never see the passphrase):
SSH_AUTH_SOCK=/run/user/$(id -u)/openssh_agent ssh-add -t 8h ~/.ssh/<your_gpu_host_key>
```

## 3. Procedures

| Task | Command (from the repo root in WSL2) | Notes |
|---|---|---|
| Push code | `scripts/gpu/sync.sh` | Mirrors the working tree; keeps the remote `.venv` and results |
| Push code for evidence runs | `SYNC_FROM=head scripts/gpu/sync.sh` | Exactly the committed HEAD, so results map to a commit (no `-dirty`) |
| Read-only inspection | `scripts/gpu/remote.sh bash scripts/gpu/inspect_host.sh` | No changes on the host |
| User-level bootstrap | `scripts/gpu/remote.sh bash scripts/gpu/bootstrap_host.sh` | Idempotent; uv + Python + locked cu130 env |
| Docker + NVIDIA toolkit | `scripts/gpu/remote.sh bash scripts/gpu/setup_docker.sh` | **System-level**, owner-approved 2026-11-10. Idempotent, never replaces existing installs. On this host it only merged `"ip": "127.0.0.1"` into the existing `daemon.json` (backup `daemon.json.bak.<UTC>`) and restarted Docker once; reruns report "already compliant" |
| Validate environment | `scripts/gpu/remote.sh bash scripts/gpu/validate_host.sh` | Bounded. Writes and pulls back `benchmarks/results/<date>-GPU-A16x1-vultr-env-validation/` |
| Any bounded command | `GPU_TIMEOUT=600 scripts/gpu/remote.sh uv run --frozen pytest -m gpu` | Prints start/end/duration/estimated cost |

Every session's start, end, duration, purpose, and estimated cost is recorded in the portfolio `PROJECT_STATUS.md` §5.2.
GPU jobs expected to exceed 15 minutes need owner approval first (Gate B).

## 4. Rebuild / recovery

The host is disposable: everything needed to recreate it is in this repository.

1. Owner provisions a replacement GPU instance (same OS image), adds the SSH key, and restricts the firewall.
   Automation never provisions, stops, restarts, or destroys instances.
2. Update `GPU_HOST` in `scripts/gpu/host.env`. Accept the new host key deliberately (`ssh-keyscan` + fingerprint
   check against the Vultr console). `StrictHostKeyChecking=yes` refuses unknown keys.
3. `scripts/gpu/sync.sh && scripts/gpu/remote.sh bash scripts/gpu/bootstrap_host.sh`
4. If container workloads are needed: `scripts/gpu/remote.sh bash scripts/gpu/setup_docker.sh`
5. `scripts/gpu/remote.sh bash scripts/gpu/validate_host.sh`, then compare `env.json` with the previous validation.

**Verified vs untested (as of 2026-11-10):**

| Step | Status | Measured |
|---|---|---|
| `bootstrap_host.sh` on the *existing* host (no uv/venv yet, warm provider network) | **Verified** | 25 s |
| `validate_host.sh` | **Verified** | 11–15 s |
| `setup_docker.sh` merge path (Docker + toolkit preinstalled) | **Verified**, idempotent rerun | 3 s |
| `setup_docker.sh` fresh-install path (no Docker on the image) | **Untested** | — |
| Full recreation (owner provisions a new instance → validated environment) | **Untested** | — |
| Parity model re-download on the host (~1.3 GB) | Observed during the bf16 run, not timed separately | — |

No end-to-end rebuild time is claimed until a complete recreation has been performed and timed. The 25 s figure covers
only our user-level bootstrap on a host whose image already had the NVIDIA driver, Docker, and the container toolkit.

**Data-loss risk on destroy:** only uncollected results and caches (uv cache, HF model cache, Docker images).
`remote.sh` pulls `benchmarks/results/` after every command, so measurements are not lost. Host-security inspection
output (`inspect.txt`) is kept locally but ignored by Git, so it never appears in a public repository.

### Snapshots (optional, not used)
Vultr charges $0.05/GB-month of compressed snapshot size. A snapshot would mainly save re-downloading wheels, weights, and
images. Support for this GPU plan has not been confirmed, and no snapshot is created without owner approval.

## 5. Billing facts that drive lifecycle decisions

- ~$0.471/hour while the instance **exists** (running or stopped). Only destroying the instance stops billing (Vultr docs).
- Destruction is recommended only with: idle days × $11.30/day saved vs rebuild time + re-download vs data at risk.

## 6. GPU hosts by profile (provider-independent)

Scripts in `scripts/gpu/` take the host from a **profile**: `GPU_PROFILE=<name>` loads
`scripts/gpu/hosts/<name>.env` (untracked and git-ignored; copy a committed `*.env.example`). Without `GPU_PROFILE`
the original `scripts/gpu/host.env` (the Vultr A16) is used. A profile sets how to reach the host, the remote repo
directory, where results live, the evidence label (`GPU_LABEL`, path-safe `GPU_RESULT_LABEL`), the hourly price,
and environment exports. Scripts branch on capabilities (is Docker available, is there persistent storage), never on
the provider. Adding a provider = one profile + `bootstrap_host.sh` + `validate_host.sh`.

| Concern | Vultr A16 (`host.env`) | RunPod A5000 (`GPU_PROFILE=runpod-a5000`) |
|---|---|---|
| Reach | `user@address` + public key of the agent identity | ssh_config alias `runpod-a5000`: address, port, key and pinned host key live in `~/.ssh/config` and `~/.ssh/known_hosts_runpod` (outside the repo) |
| Repo + Python env | `~/work/...` on the VM disk | `~/work/...` on local container disk (fast; wiped on pod restart, rebuilt by `bootstrap_host.sh`) |
| Results, logs, HF cache | VM disk | `/workspace/dli/{results,logs,hf}` on the persistent network volume (`DLI_PERSIST_DIR`, `HF_HOME`); `benchmarks/results` is a symlink |
| vLLM baseline | Docker image pinned by digest | same release from the hash-pinned venv (`bootstrap_vllm.sh`); no Docker daemon in a pod |
| CPU threads | VM cores | capped to the cgroup quota (7.65 CPUs of 96 visible) by `cpu_threads.sh` |
| Evidence label | `[GPU:A16x1@vultr]` | `[GPU:A5000x1@runpod]` |

### Connecting to a RunPod pod (owner steps, once per pod)
1. Create a dedicated key (`ssh-keygen -t ed25519 -f ~/.ssh/runpod_gpu`), load it into the agent, add the public key
   under RunPod **Settings → SSH Public Keys** (applied when a pod starts; for a running pod append it to
   `/root/.ssh/authorized_keys` from the web terminal).
2. Add a `Host runpod-a5000` block to `~/.ssh/config` with the pod's *SSH over exposed TCP* address and port,
   `IdentityFile ~/.ssh/runpod_gpu`, `IdentitiesOnly yes`, `ForwardAgent no`, `StrictHostKeyChecking yes`,
   `UserKnownHostsFile ~/.ssh/known_hosts_runpod`, `HashKnownHosts yes`. Do not use the `ssh.runpod.io` proxy
   (no rsync).
3. Read the host-key fingerprint in the pod's web terminal (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`),
   compare it with `ssh-keyscan` of the exposed port, and only then pin it. A pod restart can change the address and
   host key: repeat 2–3.

### Procedure on a fresh or restarted pod
```bash
export GPU_PROFILE=runpod-a5000
SYNC_FROM=head scripts/gpu/sync.sh
GPU_TIMEOUT=1800 scripts/gpu/remote.sh bash scripts/gpu/bootstrap_host.sh   # uv, Python, locked env, /workspace links
GPU_TIMEOUT=1800 scripts/gpu/remote.sh bash scripts/gpu/bootstrap_vllm.sh   # vLLM baseline venv (hash-pinned)
scripts/gpu/remote.sh bash scripts/gpu/validate_host.sh                     # -> <date>-GPU-A5000x1-runpod-env-validation/
```
Billing (owner-provided, 2026-11-11): ~$0.28/h while running (compute $0.27 + container disk $0.007); the 50 GB
volume is billed separately and persists when the pod is stopped. Pods are never restarted or recreated by automation.

# Operations

This runbook currently covers the **remote GPU development host**. Serving, deployment, alerting, and incident
runbooks are added as those components land (milestones M4, M8, M9).

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

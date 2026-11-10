# Security policy

## Reporting
This is a portfolio project without a production deployment. Report vulnerabilities privately through GitHub
security advisories on this repository once it is published. Please do not open public issues for vulnerabilities.

## Current security posture (M0/M1)
- No network service is exposed yet (the API server arrives in M4). The threat model will be maintained in `docs/security.md`.
- Model weights are loaded **only** from `safetensors` files (no pickle deserialization) at pinned Hugging Face revisions,
  with strict tensor name and shape validation.
- Dependencies are fully pinned in `uv.lock` and audited with `pip-audit` for both PyTorch builds. Security floors for
  transitive dependencies are declared in `pyproject.toml` (`constraint-dependencies`).
- Secrets are never committed (gitleaks pre-commit hook and CI scan).
- GPU development host: no credentials are stored there, SSH agent forwarding is disabled, and services bind to localhost
  only (`docs/operations.md`).

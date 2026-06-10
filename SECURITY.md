# Security Policy

CCO is an anti-cheat-critical project: the value of the competition rests on the integrity of the
locked harness (`benchmark.py`), the enforcement package (`cco/`), the oracles
(`references/`, `kernel_configs/`), and the manifest (`manifest.json`). A bypass of any of these
is a security issue here, even when it wouldn't be one in an ordinary library.

## Reporting a vulnerability

**Do not open a public issue or PR for a security problem.** Use GitHub's
[private vulnerability reporting](https://github.com/zeokin/Cuda-Compute-OSS/security/advisories/new)
("Report a vulnerability" under the repository's **Security** tab). Include:

- the attack: what a submission could do, and which gate/check it gets past;
- a minimal proof-of-concept `kernel.py` (or payload) demonstrating it;
- the environment if relevant (GPU, driver, torch/triton versions).

You should get an initial response within a few days. Please allow the fix to land before
disclosing publicly — a published bypass is immediately exploitable by every participant.

## What counts as a vulnerability

- **No-delegation bypass** — a way to have `torch.matmul` / `F.*` / `aten::*` / a vendor library
  do the computation that gets past *both* the static AST guard (`cco/guard_kernel.py`) and the
  runtime dispatch trap (`cco/dispatch_trap.py`).
- **Scoring manipulation** — defeating buffer rotation, the fused correctness re-check, the
  alias guard, or the seed derivation (`cco/seed.py`) to score better than the kernel deserves.
- **Manifest / integrity escape** — modifying or shadowing locked code in a way
  `cco/manifest_tool.py` verification does not flag (e.g. import-order tricks, injected files).
- **Score-blob forgery** — producing a valid-looking bound blob (`cco/blob.py`) that does not
  correspond to a real run of the locked harness.
- **Sandbox escape** — a submission that executes outside the kernel contract during scanning or
  scoring (the scanner must never execute the submission; the rerun is egress-closed).

Leaked credentials (e.g. a token in git history) are also in scope: report privately first — a
leaked token must be revoked, not just rotated.

## What is usually not a vulnerability

- Making a kernel *slower* or failing your own correctness run.
- Crashes of the harness on malformed local input (file a normal bug).
- Wins within the rules — a faster correct Triton kernel is the point of the competition.

## Supported versions

Only the current `main` branch is supported. The competition's canonical rerun always executes
the locked code at `main` plus the submitted `kernel.py`; older checkouts are not patched.

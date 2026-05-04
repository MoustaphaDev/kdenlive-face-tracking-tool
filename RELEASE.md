# Release Checklist

Use this checklist before publishing a GitHub release and sharing the repo publicly.

## 1) Scope and Version

- [ ] Confirm release scope and changelog notes.
- [ ] Bump `version` in `pyproject.toml`.
- [ ] Ensure README examples match current CLI flags and defaults.

## 2) Local Validation

- [ ] Run unit tests:

  ```bash
  uv run python -m unittest discover -s tests
  ```

- [ ] Verify CLI help:

  ```bash
  uv run kdenlive-face-mask --help
  ```

- [ ] Build artifacts:

  ```bash
  uv build
  ```

## 3) Platform and Provider Validation

- [ ] Confirm provider self-check output on at least one target machine per provider family you claim in release notes:

  ```bash
  uv run python -c "import onnxruntime as ort; print('\\n'.join(ort.get_available_providers()))"
  ```

- [ ] Validate at least one real-world run in CPU mode.
- [ ] Validate GPU mode(s) you claim in release notes (`cuda`, `rocm`, `coreml`, `openvino`) on real hardware when available.

## 4) CI and Repository Hygiene

- [ ] Confirm CI is green on Linux, macOS, and Windows.
- [ ] Review open issues/PRs for blockers.
- [ ] Confirm `LICENSE` is present and matches repo intent.
- [ ] Ensure no local secrets or machine-specific files are tracked.

## 5) Publish On GitHub

- [ ] Create and push git tag for release.
- [ ] Create a GitHub Release with concise notes and usage links.
- [ ] Include compatibility caveats in release notes:
  - CPU fallback is universal.
  - GPU providers depend on host drivers/runtime availability.
  - ROCm is Linux-focused.

## 6) Post-Release Smoke Check

- [ ] Clone the repo in a clean environment.
- [ ] Run `kdenlive-face-mask --help`.
- [ ] Perform one file-based end-to-end mask generation run.

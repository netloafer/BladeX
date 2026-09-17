# Release process

How a BladeX version gets from a commit to a git tag, a GitHub Release and (eventually)
PyPI — and which step has a gate that stops a wrong release.

This page exists because the first two releases (0.1.0, 0.2.0) shipped with **no** gate
between "tag pushed" and "wheel uploaded": 0.2.0 was tagged on a tree whose
`pyproject.toml` files still said `0.1.0`, and the only reason a wrong package was not
uploaded was that the PyPI publisher had not been configured yet. PyPI version numbers
cannot be overwritten or reused, so that upload would have been irreversible.

## 1. Before tagging

| Step | Command | Gate |
|---|---|---|
| Bump the version in **all four** `pyproject.toml` files (root + `packages/*`) | edit `version = "X.Y.Z"` | `test_release_readiness.py::test_pyproject_versions_are_identical` — any one of the four drifting turns red |
| Re-sync the editable install so `bladex --version` reports the new number | `uv sync` | `test_dunder_version_derives_from_pyproject` — stale dist metadata turns red |
| Update the version labels in `README.md` / `README.zh.md` (`(vX.Y.Z)` headings) | edit | `test_readme_version_labels_match_package_version` |
| Lint + tests | `ruff check .` then `pytest -q` (private tree: `bash scripts/gate_check.sh`, which runs both) | CI runs the same two commands on every push to `main` |
| Dry-run the tag gate locally | `python scripts/check_release_version.py --tag vX.Y.Z` | exit 0 = the tag you are about to push matches the package version |

`__version__` in `bladex_proxy` / `bladex_mcp` is **derived** from `pyproject.toml`
(installed metadata first, the adjacent `pyproject.toml` when running from a source tree).
Do not add a second literal.

## 2. Tag push → `release.yml`

```
git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z
```

The workflow runs, in order:

1. **`Check tag == package version`** — `scripts/check_release_version.py --tag <tag>`.
   The tag minus its leading `v` must equal, byte for byte, the version in all four
   `pyproject.toml` files. Mismatch ⇒ the job fails **before** anything is built.
2. `uv build --all-packages`.
3. **`Publish to PyPI`** — runs **only** when the repository variable
   `BLADEX_PYPI_PUBLISH` is exactly `true`. Otherwise the step is skipped with a notice.
   Publishing is irreversible, so it is opt-in, never the default.
4. `GitHub Release (draft)` with the built distributions attached. Publish the draft by hand.

Until the trusted publisher below is configured, leave `BLADEX_PYPI_PUBLISH` unset.

## 3. Configuring PyPI trusted publishing (one-time, done on pypi.org)

BladeX publishes three distributions: `bladex-core`, `bladex-proxy`, `bladex-mcp`.
Each needs its own publisher entry. No API token is involved; PyPI verifies the GitHub
OIDC token minted by the workflow. The claims it checks (as seen in the failed 0.2.0 run):

| PyPI field | Value |
|---|---|
| Owner | `netloafer` |
| Repository name | `BladeX` |
| Workflow name | `release.yml` |
| Environment name | *(leave blank — the job does not use a GitHub environment)* |

Steps:

1. Sign in to pypi.org → **Your account → Publishing** (`https://pypi.org/manage/account/publishing/`).
2. For a distribution that does **not** exist on PyPI yet, add a **pending publisher** with the
   project name (`bladex-core`, then `bladex-proxy`, then `bladex-mcp`) and the four fields above.
   The first successful upload creates the project and converts the pending publisher into a
   normal one. For a project that already exists, do the same under
   *Project → Manage → Publishing*.
3. Repeat on **test.pypi.org** for the rehearsal below (separate account, separate publishers).
4. Set the repository variable: GitHub → Settings → Secrets and variables → Actions →
   **Variables** → `BLADEX_PYPI_PUBLISH` = `true`. Do this only after the rehearsal passes.

## 4. TestPyPI rehearsal (do this before the first real upload)

The workflow has a manual trigger for exactly this:

1. Push the release tag as usual (the tag-push run will build and skip publishing as long as
   `BLADEX_PYPI_PUBLISH` is not `true`).
2. GitHub → Actions → **Release** → *Run workflow* → choose the **tag** `vX.Y.Z` as the ref,
   `target = testpypi`, run.
3. The run must pass the version gate, build, and upload to `https://test.pypi.org/`.
   Verify with `pip install --index-url https://test.pypi.org/simple/ --no-deps bladex-proxy==X.Y.Z`.
4. A TestPyPI version, like a PyPI one, is consumed forever — if the rehearsal must be repeated,
   use a post-release tag (`vX.Y.Z.post1` with matching `pyproject.toml` versions), not the same one.
5. Only after the rehearsal passes: configure the pypi.org publishers (§3) and set
   `BLADEX_PYPI_PUBLISH=true`. The next tag push publishes for real; `target = pypi` on the
   manual trigger does the same for a tag that was pushed while the variable was unset.

## 5. Known gaps (recorded, not fixed here)

- `actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v3` emit a
  *Node.js 20 deprecated* warning. Bumping them is a separate change (major-version bumps of
  actions can break the workflow) and should not ride along with a release.
- The GitHub Release is created as a draft with auto-generated notes; the hand-written release
  notes are pasted in when publishing the draft. Nothing checks that they exist.

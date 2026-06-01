# Releasing

YAAB uses a two-branch model so development never blocks releasing:

| Branch | Role |
|---|---|
| `develop` | Default branch. Fast-moving integration: every PR merges here. |
| `main` | Release branch. Only receives merges from `develop` when cutting a release. Every release tag points to a commit on `main`. |

Publishing is fully automated through PyPI **Trusted Publishing** (OIDC) — no
API tokens exist anywhere. The trusted publishers are configured on PyPI for
the `yaab-sdk` and `yaab-core` projects, both pointing at this repository's
`release.yml` workflow.

## Cutting a release

1. **Make sure `develop` is green** (CI passing on the merge you want to ship).

2. **Set the version.** On `develop`, `pyproject.toml` `[project] version` and
   `yaab-core/Cargo.toml` + `yaab-core/pyproject.toml` versions must be the
   version you are about to release, and `CHANGELOG.md` must have a section for
   it. (Do this in a normal PR like any other change.)

3. **Fast-forward `main` to the release point:**

   ```bash
   git checkout main
   git pull origin main
   git merge --ff-only origin/develop   # or merge a specific commit
   git push origin main
   ```

4. **Tag and push** (this is the trigger):

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. `release.yml` then:
   - **preflight** — refuses the tag unless its commit is on `main` AND the tag
     matches `pyproject.toml`'s version;
   - builds the `yaab-sdk` sdist + wheel, runs `twine check`, and smoke-tests the
     wheel in a clean venv (`import yaab` + `yaab info`);
   - builds `yaab-core` abi3 wheels for Linux x86_64/aarch64, macOS
     x86_64/arm64, and Windows x64;
   - publishes both projects to PyPI via OIDC;
   - creates a GitHub release with the artifacts and auto-generated notes.

## Naming

- PyPI distribution: **`yaab-sdk`** — the import package and CLI are **`yaab`**.
- Rust accelerator: **`yaab-core`** (optional; `pip install 'yaab-sdk[rust]'`).

## If a release goes wrong

- A failed workflow publishes nothing (publish jobs run only after build + smoke
  test succeed). Fix the problem on `develop`, merge to `main`, delete the tag
  (`git push origin :refs/tags/vX.Y.Z`), and re-tag.
- PyPI does not allow re-uploading a version that already published. If the bad
  artifact reached PyPI, bump the patch version and release again ("yank" the
  bad version on PyPI if it is harmful).

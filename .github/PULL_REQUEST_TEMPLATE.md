## What & why

<!-- What does this change do, and why? One or two paragraphs. -->

## Risk areas touched

- [ ] Storage pipeline (Pipeline / Memory Ledger / Memory Index) - if yes, rebuild-equivalence verified?
- [ ] Injection / prefetch (memory that reaches the model)
- [ ] Routing
- [ ] Identity / auth / binding
- [ ] None of the above

## Verification

- [ ] `uv run ruff check .` clean
- [ ] `uv run pytest -q` passes
- [ ] New behavior has a test
- [ ] No `print()` debugging; structlog / stdlib logging used appropriately

## Notes for reviewers

<!-- Anything load-bearing, tricky, or worth a second look. Link related issues. -->

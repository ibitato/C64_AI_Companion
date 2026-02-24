# Documentation Style Guide

## Language

- All maintained documentation must be in English.
- Use direct, technical wording.
- Avoid marketing language and vague claims.

## Structure

Use this order when applicable:

1. Purpose
2. Preconditions
3. Procedure
4. Validation
5. Failure modes and recovery
6. References

## Command Quality

- Commands must be copy-paste ready.
- Use repository-relative paths.
- Keep command examples consistent with actual scripts and flags.

## Consistency Rules

- Use `models/Ministral-3-8B-Thinking` when referencing base model path.
- Use `docker compose` syntax consistently.
- Prefer explicit paths over implied context.

## Maintenance Rules

- Update docs in the same change as behavior changes.
- Remove stale instructions instead of keeping contradictory legacy text.
- Keep compatibility stubs only when a path rename is required.

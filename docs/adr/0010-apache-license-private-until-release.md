# ADR-0010: Apache-2.0 license; personal GitHub; private until release

- Date: 2026-09-27
- Status: Accepted

## Context

The project intends to be an open benchmark, but early results, prompts and scenario designs should not leak before they are validated.

## Decision

The code is licensed Apache-2.0 (LICENSE at the repository root). It is hosted on a personal GitHub account and remains private until an explicit release ADR is accepted.

## Alternatives rejected

Public from day one (premature exposure of unvalidated results). A copyleft license (friction for benchmark adopters). An organization account (unnecessary administration at this stage).

## Consequences

No public repositories, gists, releases, packages or PyPI uploads, and no posting of code or results to external services, until the release ADR. Tools that would do any of these must stop and ask.

## Status

Accepted 2026-09-27.

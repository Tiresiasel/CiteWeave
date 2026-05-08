# Manual / integration scripts

These files are intentionally excluded from default `pytest` discovery.
They exercise local services, real processed papers, external LLM flows, or
mutable runtime configuration, so they are not reliable unit tests.

Run them explicitly only after starting the required local services and setting
runtime configuration in `.env` / `config/*.local.json`.

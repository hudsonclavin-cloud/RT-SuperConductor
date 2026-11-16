# MCP connectors for Claude Code

Project-level servers are declared in `/.mcp.json`. They are optional tooling for
interactive investigation; the pipeline itself does not depend on them.

| Server | Purpose | Status when this was added |
| --- | --- | --- |
| `materials-project` | Search/fetch Materials Project records (`mpmcp`, shipped in `mp-api[mcp]`, Python >= 3.12) | Launch verified (`mpmcp --help`); live tool calls not tested |
| `openalex` | Literature search and citation traversal (hosted MCP) | Not reachable from the cloud environment until `mcp.openalex.org` is allowed; REST API at `api.openalex.org` works |
| `playwright` | Browser interaction/testing | Package exists on npm; not exercised |
| `context7` | Current library documentation | Package exists on npm; needs `context7.com` egress; not exercised |

## Policy

- Connectors must not bypass the existing distinctions: the ML phonon screen is
  diagnostic and excluded from candidate ranking.
- Literature found through OpenAlex goes into a review queue (DOI, sample identity,
  pressure, temperature range, method, exact location; unknown stays unknown).
  It never updates training labels without the project's review gate.

## Keys

`MP_API_KEY` is read from the environment (same variable the pipeline uses). In
cloud sessions the proxy injects it for `api.materialsproject.org`; locally,
export a real 32-character key. Never commit keys.

## Not included

- Aster MCP (`npx -y aster-mission-mcp`) belongs to the asteroid-planner repo.
- Zotero, SciX, Weights & Biases, Hugging Face: they need personal accounts/tokens;
  add only when a concrete workflow needs them.

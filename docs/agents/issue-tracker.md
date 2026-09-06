# Issue tracker: Multica (primary) + GitHub (public)

Two surfaces, different jobs. Don't file the same thing twice.

| Surface | Role | Tool |
|:--|:--|:--|
| **Multica** | Primary execution environment. Where work is planned, assigned to agents, and run. | `multica` CLI |
| **GitHub** (`27b-io/alaya`) | Public commentary. External bug reports, discussion, anything a non-workspace reader must see. | `gh` CLI |

Multica project: `alaya` (`2d8f6caa-d432-4cf2-ab44-1b0918ff5cfa`). Issue identifiers are
`LAB-NNNN` — the same identifiers that appear in commit subjects and PR titles. The GitHub
repo is **not** registered in Multica's repo list, so nothing mirrors automatically; the
`LAB-NNNN` reference in a commit or PR body is the only link.

## When a skill says "publish to the issue tracker"

Create a Multica issue:

```bash
multica issue create \
  --title "..." \
  --description-stdin \
  --project 2d8f6caa-d432-4cf2-ab44-1b0918ff5cfa \
  --status backlog
```

**Default to `--status backlog`.** `todo` is a fire gate: moving an issue to `todo` starts
an agent run against its assignee. See the `placing-multica-issues` skill for the full
backlog-vs-todo rules — don't re-derive them here.

Only open a GitHub issue instead when the item is genuinely for a public audience.

## When a skill says "fetch the relevant ticket"

- `LAB-NNNN` → `multica issue get LAB-NNNN`, plus `multica issue comment list LAB-NNNN`
- bare `#NN` → GitHub: `gh issue view NN --comments`

## Conventions

- **List**: `multica issue list --project 2d8f6caa-d432-4cf2-ab44-1b0918ff5cfa --status <key> --output json`
- **Search**: `multica issue search "<text>" --output json`
- **Comment**: `multica issue comment add LAB-NNNN --body "..."`
- **Label**: `multica issue label add LAB-NNNN <label>` / `label remove`
- **Status**: `multica issue status LAB-NNNN <key>` — keys: `backlog`, `todo`,
  `in_progress`, `in_review`, `done`, `blocked`, `cancelled`. Add `--no-start` to change
  status without launching an agent.
- **Assign**: `multica issue assign LAB-NNNN --to <member|agent|squad>` (also takes
  `--no-start`).
- Multi-line descriptions: pipe via `--description-stdin`. Never `--description` with
  embedded newlines; it decodes escapes and mangles content.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if external PRs on `27b-io/alaya` should
enter the triage queue.)_

## Wayfinding operations

Used by `/wayfinder`. Multica models this natively — no synthetic map issue needed.

- **Map**: a parent Multica issue.
- **Child ticket**: `multica issue create --parent <map-id> [--stage N]`. `--stage` groups
  children into an ordered barrier; the parent's assignee wakes only when a whole stage
  finishes.
- **Blocking**: `blocked` status. `multica issue children <map-id>` lists children grouped
  by stage.
- **Frontier**: children in `backlog`/`todo` with no assignee, lowest stage first.
- **Claim**: `multica issue assign <id> --to <you>`.
- **Resolve**: comment the answer, then `multica issue status <id> done`.

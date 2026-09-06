# Triage Labels

The skills speak in terms of five canonical triage roles. This repo tracks work in
**Multica**, which expresses three of them as labels and two as native state.

| Label in mattpocock/skills | In our tracker (Multica)                      | Meaning                                  |
| -------------------------- | --------------------------------------------- | ---------------------------------------- |
| `needs-triage`             | label `needs-grooming` (status `backlog`)     | Maintainer needs to evaluate this issue  |
| `needs-info`               | label `needs-info`                            | Waiting on reporter for more information |
| `ready-for-agent`          | label `ready-for-todo`, then status `todo`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | assign a **member**; status `todo --no-start` | Requires human implementation            |
| `wontfix`                  | status `cancelled`                            | Will not be actioned                     |

## Why two roles have no label

- **`ready-for-human`**: Multica already encodes agent-vs-human in `assignee_type`
  (`agent`/`squad` vs `member`). A parallel label would be a second source of truth that
  can disagree with the assignee.
- **`wontfix`**: `cancelled` is a real board state that removes the issue from every
  active query. A label would leave it sitting in `backlog` looking live.

Moving an issue to `todo` **starts an agent run**. Applying `ready-for-todo` is the safe,
non-firing signal; the status change is the commitment. Pass `--no-start` when you want
the column without the launch.

If a skill needs a label that doesn't exist here, prefer the native state above over
`multica label create`.

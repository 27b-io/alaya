# Triage Labels

The skills speak in terms of five canonical triage roles. This repo tracks work in
**Multica**, which expresses three of them as labels and two as native state.

| Label in mattpocock/skills | In our tracker (Multica)                      | Meaning                                  |
| -------------------------- | --------------------------------------------- | ---------------------------------------- |
| `needs-triage`             | label `needs-grooming` (status `backlog`)     | Maintainer needs to evaluate this issue  |
| `needs-info`               | label `needs-info`                            | Waiting on reporter for more information |
| `ready-for-agent`          | label `ready-for-todo`, assign an **agent**/squad, then status `todo` | Fully specified, ready for an AFK agent |
| `ready-for-human`          | assign a **member**; status `todo --no-start` | Requires human implementation            |
| `wontfix`                  | status `cancelled`                            | Will not be actioned                     |

## Why two roles have no label

- **`ready-for-human`**: Multica already encodes agent-vs-human in `assignee_type`
  (`agent`/`squad` vs `member`). A parallel label would be a second source of truth that
  can disagree with the assignee.
- **`wontfix`**: `cancelled` is a real board state that removes the issue from every
  active query. A label would leave it sitting in `backlog` looking live.

## The fire gate

Moving an issue to `todo` **starts an agent run — but only if an agent or squad is
assigned.** The status is not the trigger on its own: the board routinely holds `todo`
issues assigned to members, and unassigned ones, and neither fires anything. So
`ready-for-agent` is a two-part commitment — the assignee names *who* runs, the status
says *now*.

Applying `ready-for-todo` is the safe, non-firing signal. Pass `--no-start` to either
`multica issue status` or `multica issue assign` when you want the column or the owner
without the launch.

## When a role isn't in the table

Map it to a native state **explicitly**, or stop. Never substitute whichever state looks
closest: `todo` can launch an agent run and `cancelled` drops the issue out of every
active query, so a wrong guess either burns a runtime or silently buries work.

With no mapping, ask for a triage decision. Don't reach for `multica label create` —
a new label is a new source of truth, which is the thing this file exists to prevent.

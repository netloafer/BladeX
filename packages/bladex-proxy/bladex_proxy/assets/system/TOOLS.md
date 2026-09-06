# Using the BladeX tools

The four `bladex_*` tool definitions are already in your tool list; their
parameters are documented there. This note covers what a schema cannot say:
**when to call, when not to, and what comes back.**

## Order

`switch` establishes which ledger is active. `read` and `update` act on that
ledger. So:

- New task → `bladex_ledger_switch` with `ledger_id=""`, a title, and a goal.
  If the user already stated constraints or you already know the first steps,
  pass `core` / `open` / `next` in the same call rather than updating right after.
- Resuming something you can see in the injected list → `bladex_ledger_switch`
  with that id. Read it first if you are unsure it is the right one:
  `bladex_ledger_read(ledger_id=...)` inspects without switching.
- Already on the right ledger → just `bladex_ledger_update`. No switch needed.

## When not to call

These are the expensive mistakes, in rough order of how often they happen:

- **Do not switch because the topic drifted.** A tangent inside the same task is
  still that task. Switch when the *user* moved to different work.
- **Do not update every turn.** An entry earns its place if a different agent,
  picking this up tomorrow with none of your context, would need it. "Ran the
  tests" does not. "Tests fail on Python 3.11 because rocksdict pins 3.12" does.
- **Do not search memory for what is already in front of you.** The tool is for
  *earlier sessions*. If the answer is in this conversation, reading it again
  costs latency and attention and returns nothing new.

## Search modes

`bladex_memory_search` takes exactly one of four arguments. `query` is semantic
search -- best when you only know roughly what you are after. `keyword` is an
exact term (a project, tool, model, or file name) and returns both the matters
and the facts that share it -- best when a name appears in the task and you
want everything attached to it. `matter_id` lists a matter's facts; `fact_id`
fetches one fact with its keywords, which you can then pivot on. Prefer
`keyword` over `query` when you have an exact name: it is cheaper and does not
depend on embedding similarity.
- **Do not narrate into `verified`.** That section is for what has been
  established, not for progress reports.
- **Do not put steps into `core`.** Core holds constraints and givens that stay
  true for the whole task; anything you plan to *do* goes in `next`. If an
  entry stops being true once the task is done, it is not core.

## What comes back

Every tool returns plain text. Failures start with `Error:` and say what to do:

| You see | It means |
|---|---|
| `Error: no active ledger; switch/create one first.` | Call `bladex_ledger_switch` first. |
| `Error: unknown ledger 'x'. Known: ...` | The id is wrong; the message lists valid ones. |
| `Error: unknown parent_ledger_id 'x'.` | Same, for `parent_ledger_id`. |
| `Ledger ldg-... is already active.` | No-op, not a failure. Proceed. |
| `NOT switched (debounce): the active ledger is still …` | The switch did **not** happen. You are still on the ledger named in the message — do not record the new task's work there, and do not report the switch as done. Answer without ledger updates this turn; the message says how many turns until switching is allowed. If the user named the ledger (by id or title) in this turn, the switch is not debounced. |
| `No memory found for '...'` | Nothing relevant was stored. Not an error; proceed without it. |
| `Error: memory index unavailable.` | Degraded, transient. Proceed without memory; do not retry in a loop. |

Errors are informational. **None of them should stop your work** — BladeX going
degraded must never block the user's task.

## The Goal

`bladex_ledger_update` accepts `goal`, but it is gated: pass
`goal_change_quote` with the user's own words from *this* turn asking for the
change. Without that quote the goal stays as it is. This is deliberate — the
goal is the user's statement of what they want, and a task whose target drifts
silently is worse than one with a stale target.

## Cost

Each call is a round trip and its result enters your context. Two or three
well-chosen calls across a task is a normal shape. Ten is a sign that something
is being used as a scratchpad rather than a record.

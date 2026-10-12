# Research workflow guide (for the human)

[← Documentation index](README.md)

This page is for you, not the agent. It explains how the pieces fit, what to do in
each Gemini session, and how to catch up after a break. The agent's rules live in
`AGENTS.md` and `GOAL.md`; you rarely need to reread them.

## 1. The picture in five lines

- **Goal:** a curation method that rejects more bad clips without losing good
  speech. It is proven on clips **you** label, then shown to improve a TTS model.
- **Paper:** Interspeech 2027, deadline ≈ early Feb 2027.
- **Your role:** decide, listen, read the core papers, approve changes to the goal,
  write the paper.
- **Gemini 3.8 Flash's role:** implement, run experiments, keep the records, and
  propose next steps.
- **Where we are:** always in `research/STATUS.md`.

## 2. Where things live

| File | What it is | Who writes it | When you read it |
|---|---|---|---|
| `research/STATUS.md` | Current state on one page: verified facts, unverified assumptions, open decisions, test-set uses | Agent, rewritten each session | **Every time you come back** |
| `GOAL.md` | The contract: success criteria, forbidden shortcuts, session rules, milestones | You approve every change | When the agent proposes a change |
| `WORKLOG.md` | Diary, oldest at top, newest at bottom | Agent, one entry per session | Last 2–3 entries when catching up |
| `docs/worklog_archive/` | Older diary months | Agent, monthly | Only when searching history |
| `research/GRAND_PLAN.md` | Menu of planned experiments | You + agent | When choosing the next task |
| `research/frozen_artifacts.sha256` | Fingerprints of data that must never change (gold set, 300 anchors, later your test set) | You approve | Never; it is checked by command |
| `.data/goal/review/<topic>/` | Listening batches for you (`REVIEW.md`, `review.csv`, `playlist.m3u`) | Agent | When the agent asks you to listen |
| `docs/experiments.md` | What was measured before October 2026 | — | Background |

## 3. One Gemini session, step by step

**Start.** Paste this, filling in one task:

```text
Read AGENTS.md, GOAL.md and research/STATUS.md, then do the GOAL.md §7 start steps.
Today's single task: <one sentence, e.g. "measure the free Gemini quota with 5 calls">.
Do nothing else. If the task needs my listening or a change to GOAL.md, stop and ask.
End with the §7 report (Changed / Numbers / Deviations / Assumed, not verified).
```

Pick tasks from the "Current milestone" and "Open decisions" sections of
`STATUS.md`. If you can't phrase the task in one sentence, it's two sessions.

**During the session.** Answer its questions. If it starts something you didn't
ask for, say "stop, that's not today's task".

**End (≈ 2 minutes).**

1. Read the report. **Deviations** and **Assumed, not verified** are the two lines
   that matter.
2. Look at what it committed:
   ```bash
   git log --oneline -5
   git show --stat HEAD
   ```
3. Check for red flags (§4). If anything looks wrong, tell it in the next session:
   "Commit X did Y, which breaks GOAL.md §N. Revert or justify."
4. Start a **new** session for the next task. Don't continue a long conversation;
   that is where the agent forgets its rules.

## 4. Red flags when reviewing a session

| You see | Why it matters |
|---|---|
| A new `.html` file for listening or review | Breaks GOAL §6; can mismatch clips and labels. |
| A new `test.py`, `tests/`, or test runs you didn't ask for | Against the repo rules. |
| Changes to `GOAL.md`, `research/frozen_artifacts.sha256`, or a failed hash check | Goal or data drift. Only you approve these. |
| A number without a file path, or "should", "likely", "expected" stated as a result | Belief drift. Ask for the file. |
| A new entry under "Test-set evaluations" in `STATUS.md` | Each one uses up the test set. Ask why. |
| Agreement with Gemini presented as an improvement | Gemini labels are not ground truth (GOAL §2). |
| Many Gemini calls, or "quota exhausted" followed by retries | The quota is shared and limited (GOAL §5). |
| A big result (e.g. > 90 % detection) with no "alternative explanations" | Treat it as a bug until checked (GOAL §3). |
| `commit --no-verify` or `GOAL_CHANGE_APPROVED` in its commands | It bypassed your approval. |

## 5. When the agent asks you to listen

1. Open `.data/goal/review/<topic>/REVIEW.md`. It says what to listen for and how
   long the batch takes (≤ 15 min).
2. Play the batch:
   ```bash
   mpv --playlist=.data/goal/review/<topic>/playlist.m3u
   ```
   To hear one clip's ending only: `mpv --start=-1 <path>`.
3. Fill in `your_label` (and optionally `your_note`) in `review.csv` with any
   spreadsheet app. Don't reorder rows or edit other columns.
4. Tell the agent "labels done for `<topic>`" in a new session.

Listen with headphones, in one sitting, without looking at any model output first.

## 6. Approving a change to the goal

The agent can only propose changes to `GOAL.md` (in `WORKLOG.md`). If you agree,
make or accept the edit yourself, then commit with:

```bash
GOAL_CHANGE_APPROVED=1 git commit -m "docs: <what changed and why>"
```

Without that variable, the local hook refuses the commit. On a new machine,
install the hook once:

```bash
ln -sf ../../research/pre_commit_goal_guard.sh .git/hooks/pre-commit
```

## 7. Catching up after a break

```bash
cat research/STATUS.md                                           # where we are
git log --oneline --since="7 days ago"                           # what happened
grep -n '^## ' WORKLOG.md | tail -10                             # recent diary titles
sha256sum --quiet -c research/frozen_artifacts.sha256; and echo "frozen data OK"
```

(The last line uses fish syntax; in bash use `&& echo`.) Then compare "Current
milestone" with today's date against the milestone table in `GOAL.md` §1.

## 8. Weekly (≈ 30 minutes)

1. **Milestone check.** Are you on track for the next date in `GOAL.md` §1? If not,
   decide now (cut scope or switch to the fallback paper). Don't let the agent
   drift into it.
2. **Outside audit.** Ask a different model (e.g. Claude) once:
   ```text
   Read GOAL.md, research/STATUS.md, the WORKLOG.md entries since <date> (bottom-up,
   not the whole file) and `git log --since=<date> --stat`. List every departure
   from GOAL.md with the commit or file as evidence. Do not change anything.
   ```
3. **One paper.** Read the method and results of one core paper yourself and add
   two lines to `research/README.md` on what it supports.
4. **Listen freely** to 10–20 pipeline clips with no task in mind. Most good ideas
   in this project came from hearing real failures.

## 9. Things only you do

- Choose the G1 test recordings and label them, with a second annotator.
- Decide the paper's main claim, and switch to the fallback if the 2026-12-01
  milestone is missed.
- Approve any change to the goal or to frozen data.
- Write the introduction, method and conclusions, and check the venue's AI-use
  disclosure policy.
- Spend money (paid API credit).

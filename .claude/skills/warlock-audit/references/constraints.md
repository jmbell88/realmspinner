# Constraints — paste this block verbatim into every subagent prompt

These rules are not advice. Each one is a recorded incident in this repository.

- **There is one orchestrator and you are not it.** Do not spawn subagents, do not use the
  Agent tool, do not start another audit or another fix pass. Several explorers are reading
  this tree at the same time as you; a second layer of them would duplicate the reading,
  hide the coverage and put two writers in one file. You return records or a fix and nothing
  else: the audit file, `TODO.md`, `docs/INVARIANTS.md`, `CHANGELOG.md`, the manual and the
  commit belong to the session that launched you. A defect outside the files you were given
  is still worth reporting — point `where` at the real file and let the merge decide.
- **Git is read-only for you.** Never run `git stash`, `git checkout`, `git reset`,
  `git restore`, `git clean`, `git add` or `git commit`. On 2026-09-04 a fixer ran
  `git stash` to compare pre-fix behaviour and reverted nine other agents' edits and the
  user's uncommitted work; recovery took `git fsck`. To see the committed version of a
  file, read `git show HEAD:<path>`. To see what has changed, read `git diff -- <path>`.
- **Writing `HEAD`'s content over a working file is checking it out by hand.** Restoring
  the committed text so you can watch your regression fail, then re-applying your fix, is
  the 2026-09-04 incident in different clothes: on 2026-09-09 a fixer did it to
  `panes/clay_outliner.py` while that file carried the user's uncommitted work, which
  survived by luck rather than by method. To see how the unfixed code behaves, load the old
  source as a **throwaway module in your scratchpad** — write `git show HEAD:<path>` to a
  scratch file, import it under another name, and exercise it there; two fixers in that same
  pass did exactly this and proved the same thing without touching the tree. A file in the
  tree only ever moves forwards.
- **Never run the full test suite.** Run only the test files your brief names, and only
  as `uv run pytest <files> -n 0`. Never a bare `uv run pytest`, never `-n auto`, never
  `--dist load`, never `-m gpu`, never `-m perf`. Several tests read module source while
  they run, so a suite run while another agent edits `src/` fails both of you. The
  orchestrator runs the suite once, after everyone has returned.
- **Scratch files go in the scratchpad path your brief gives you**, never in the tree.
- **Never cite `TODO.md`, a plan file, or the audit file's name from `src/` or
  `scripts/`.** `tests/test_ux_todo_fixes.py` refuses it. Cite the programme instead:
  "the 2026-09-05 audit, finding sirens-03".
- **Uncommitted changes you did not make are the user's.** Do not revert them, do not
  tidy them, do not report them as defects unless they are one.
- **`uv run --no-sync`** for any Python you run; a bare `uv sync` prunes the extras and
  breaks collection for ten test files.
- **Windows.** Paths may have spaces; quote them. Line endings in the working copy are
  CRLF and in blobs LF; do not "fix" line endings.

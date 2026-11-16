---
name: write-agent-skill
description: Use when writing a new agent skill for cuDF contributors under .agents/skills/, or reworking an existing one. Covers when a skill is worth adding, the file layout, the description that decides when it triggers, scripts versus prose, checking it, and the PR. Not for the signed skills/ tree, which the NVSkills pipeline produces.
---

# Writing an agent skill for cuDF

A skill is a folder under `.agents/skills/<name>/` with a `SKILL.md`. An agent reads every skill's `name` and `description`, and loads the body only when the description matches the task. So the description decides whether the skill is ever used, and the body decides whether the agent gets the task right once it is.

`skills/` at the repository root is a different thing: user-facing skills that the NVSkills pipeline validates and signs (pre-commit excludes that tree). Do not hand-edit it or add contributor skills there.

## 1. Decide whether it should be a skill

Write a skill for a task that recurs, has steps that are easy to get wrong, and needs knowledge the code does not show: the order to build packages in, which generated files a hook rewrites, which CI job to copy. Good examples in this repository: `retarget-pr`, `upgrade-polars`, `reproduce-ci`.

Do not write one when:
- an existing skill covers it. Read `.agents/skills/*/SKILL.md` first and extend or link to that skill instead (for example, link to `build-test-cudf` for builds and `reproduce-ci` for CI failures, rather than repeating their steps);
- it is general advice an agent already follows (write tests, read the code first);
- it is your personal workflow. Reviewers ask for those parts to be removed (Git habits, issue drafting, local paths, your editor). Keep only what every cuDF contributor needs for this task.

Before writing, do the task once by hand or with an agent and note where it went wrong. Those mistakes are what the skill is for.

## 2. Layout

```
.agents/skills/<name>/
  SKILL.md            required
  <script>.sh|.py     optional: steps that must be exact
  references/*.md     optional: long material loaded only when needed
```

- `<name>` is lowercase words joined by hyphens, verb first, and ends with the area when that is not obvious: `build-test-cudf`, `debug-cudf-pandas`, `update-pandas-testing`. The frontmatter `name` must equal the folder name.
- Keep `SKILL.md` short enough to read in one go (the existing ones run from about 30 to 470 lines; under 200 is a good target). Move long background, histories and per-version tables into `references/` and say in `SKILL.md` when to open each file.

## 3. The frontmatter

```yaml
---
name: <folder-name>
description: Use when <the task, in the words a user would use>. Covers <the main steps>. Not for <the nearby task it must not catch>.
---
```

The description is the only part that is always in context, so:
- start with when to use it, in the words people actually type ("retarget a PR to release/26.12", "pandas test failures under cudf.pandas");
- say what it is not for when a nearby request would match. A reviewer pointed out that "rebase a PR onto a branch" also caught a plain "rebase my PR on main";
- one to three sentences, no more than 1024 characters, no version numbers or dates that will go stale.

## 4. The body

Write for an agent that is capable but has never seen this repository.

- **Steps in order**, each with the exact command and how to tell whether it worked. Say what to do when it fails: stop and report, ask the user, or retry.
- **Prerequisites first**, each with a check (`gh auth status`, `docker info`). Say what to tell the user when one is missing. Never have the agent print tokens or run `gh auth login` itself.
- **Outward-facing actions need the user.** Pushing, force-pushing, editing a PR or issue, posting a comment: print the command for the user, or run it only when the user asked for it.
- **Explain the reason** for any rule that looks arbitrary, in one clause. An agent follows a rule it understands and applies it to cases the rule did not list.
- **Point at sources instead of copying them.** Link to `dependencies.yaml`, the CI workflow or the contributing guide rather than pasting their contents, and do not hard-code Python, CUDA or package versions; read them from the repository. Copied facts go stale (see #25280, which fixed stale references found by an audit).
- **Generated files**: if a pre-commit hook rewrites a file (for example `rapids-dependency-file-generator`), tell the agent to edit the source and run the hook, not to edit the output.
- **No personal paths, hostnames, credentials or internal links.**

## 5. Scripts versus prose

Put a step in a script when the commands must be exactly right, when there are several of them in sequence, or when a mistake is expensive (rewriting history, deleting files). `retarget-pr` moved its rebase into `retarget.sh` after review found flags that did not exist (`git switch --autostash`) and paths that could not complete (signing on the unsigned path).

A script in a skill should:
- run from a cuDF checkout and fail with a clear message otherwise;
- never push, edit a PR or delete the user's work; print the follow-up commands instead;
- exit with distinct codes the `SKILL.md` maps to actions (0 done, 1 usage error, 2 needs the user, and so on);
- be executable and pass the repository's pre-commit hooks (shellcheck, ruff, verify-copyright). Python scripts mark each `print` with `# noqa: T201`, as the existing skill scripts do.

Keep judgement in the prose: which option to pick, what to ask the user, how to read the output.

## 6. Check it

1. Run `python .agents/skills/write-agent-skill/check_skill.py .agents/skills/<name>` (or pass `.agents/skills` to check every skill). It checks the frontmatter, the folder name, the description length and relative link targets, and warns about repository paths in backticks that do not exist (a warning, because a skill may name build outputs or another repository's files).
2. Run `pre-commit run --files .agents/skills/<name>/*`.
3. Run every command in the skill once. Reviewers find commands that do not work.
4. Try it in a fresh agent session with no other context: one prompt that should trigger it, phrased as a user would, and one nearby prompt that should not. Read what the agent did, not only whether it finished. Fix the skill where it went wrong and try again.
5. Read it once more as a reviewer: is anything in it general advice, a personal habit, or a copy of another document?

## 7. The PR

- One skill per PR, unless a code change and the skill that uses it belong together (for example #24825 added `upgrade-polars` with the Polars 2.1 support it describes).
- In the description, say what task the skill is for, how you tried it (the prompts and what the agent did), and anything it cannot do yet.
- When a later PR changes a command, path or file that a skill names, update the skill in that PR.

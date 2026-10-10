---
name: relax-github-workflow
description: >-
  Prepare and publish Relax GitHub contributions, create or update pull requests,
  and address review feedback. Use when opening a PR, choosing a contribution branch,
  publishing changes, following up on a review, or organizing dependent Relax PRs.
  Connects commit, code-review, and CI workflows.
---

# Relax GitHub workflow

Use this workflow for contributions to `redai-studio/Relax`. Read `AGENTS.md` and `CLAUDE.md` first. For internal GitLab/GitHub synchronization, use [sync-github](../sync-github/SKILL.md). Publishing stays within the user's request; preparing a change does not itself authorize a push, PR, or merge.

## Identify the repository and branch

Inspect the working tree, remotes, authenticated account, and target repository permissions:

```bash
git status -sb
git remote -v
gh api user --jq .login
gh api repos/redai-studio/Relax --jq '{default_branch, permissions}'
```

Do not assume `origin` points to the upstream repository. Select the actual upstream and push remotes; leave unrelated files and worktrees alone. For an existing PR, verify its head repository, branch and SHA before editing or publishing an update.

- Base new contributions on the current upstream `main`, unless the task needs another base or a stack layer.
- Publish ordinary contributions from a personal fork, even with upstream write access. Push personal branches directly to `redai-studio/Relax` only when needed, such as for shared development or gh-stack.
- When pushing directly to `redai-studio/Relax`, name the branch `<name>/<type>/<description>`. Use the contributor's chosen name or GitHub login for `<name>` and the commit type for `<type>`.
- For a fork, branch names are unrestricted; `<type>/<description>` is recommended.

Fetch the chosen upstream before creating the branch. Use a separate worktree when it helps preserve ongoing work.

## Prepare and publish a PR

1. Implement the requested change and verify the affected behavior. Use [git-commit](../git-commit/SKILL.md) to stage only intended files, run pre-commit, and commit.
2. Review the diff with [code-review](../code-review/SKILL.md). Resolve evidenced issues and record relevant validation limits; a documentation-only change does not require GPU training.
3. Read `.github/PULL_REQUEST_TEMPLATE.md`. Preserve its headings and checkboxes, explain the problem, resulting behavior and validation, and mark only checks actually completed. Use a Conventional Commits title; Relax squash merges use the PR title as the resulting commit subject.
4. When publication is authorized, push to the selected remote and create the PR with explicit repository, base and head. Prepare a body file so multiline content is passed intact:

   ```bash
   git push --set-upstream <push-remote> HEAD
   gh pr create --repo redai-studio/Relax --base <base-branch> \
       --head <head-branch-or-owner:branch> --title '<type>: <description>' --body-file <body-file>
   ```

5. Read back the PR title, body, head and base. Check the published diff before advancing review; report the PR URL and verification results. For an update, verify that the remote PR head matches the intended commit.

Use a Draft PR while work is still in progress. Mark it ready when it is ready for review; Nyanpasu follows non-Draft PRs. Use [relax-github-ci](../relax-github-ci/SKILL.md) to inspect required checks and investigate failures. Required checks and the repository's review requirements determine merge readiness; an optional check failure does not by itself block merging.

## Address review feedback

Read the original review thread and the affected code. Apply reasonable findings, or explain a disagreement with evidence. Verify the fix, publish the intended update, and follow up in the original thread within the user's authorization. Use the code-review skill's follow-up rules to avoid repeating settled findings.

Nyanpasu review is triggered by PR creation or readiness, new pushed commits, replies in its existing review threads, and comments containing `/review` or `@rai-studio-bot`. Prefer the existing thread for the same issue; request another review when it is useful rather than posting a trigger after every automatic update.

## Dependent PRs

For dependent changes, consider using `gh-stack` to split the work into cohesive, independently reviewable PRs.

# Workspace Reproduction Policy

For reproduction tasks in this workspace:

- Reproduce every required step in the referenced official pipeline.
- Do not silently skip, replace, shorten, reorder, or approximate a step.
- Before treating any official step as unnecessary, explain the exact step, reason, expected impact, and ask the user for confirmation.
- Record frame ranges, frame order, prompts, thresholds, model/source revisions, environment manifests, commands or entry scripts, logs, validations, and output paths.
- Keep official-compatible results separate from corrected or extended results.
- Any user-approved substitution must be named explicitly; a compatibility directory name must not be presented as proof that its named model ran.
- Preserve failed official results and report their quality honestly instead of hiding them behind corrected output.

# Workspace Status Policy

- Do not include work from `workspace-b/` in project status.
- Do not modify or inspect `workspace/` unless the user explicitly requests it.
- Treat `workspace_whz/PROJECT_STATUS.md` as the canonical progress record.
- Treat `workspace_whz/DECISIONS.md` as the canonical architectural decision record.

## Progress maintenance

After materially changing code under `workspace_whz/`:

1. Update `workspace_whz/PROJECT_STATUS.md`.
2. Update Completed, In progress, Known issues, and Next steps.
3. Record the verified commit or current Git state.
4. Do not mark untested work as completed.
5. Ignore changes that only affect `workspace/`.

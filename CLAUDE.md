# deepclaude

Claude Code with any LLM backend — DeepSeek, Gemini 3.5 Flash, OpenRouter, Fireworks AI.

Zero npm dependencies. ESM. Node.js built-in test runner (`node:test`).

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Bugs/errors → invoke /investigate
- QA/testing → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Security audit → invoke /cso
- Parallel work → invoke /dispatching-parallel-agents
- Agent orchestration → invoke /agent-orchestration
- Brainstorming → invoke /office-hours
- Architecture → invoke /plan-eng-review

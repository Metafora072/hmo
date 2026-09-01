---
name: openchat
description: Operate this project's concise, evidence-first multi-AI collaboration workspace. Use when reading or appending daily conversations, relaying web-model replies, writing participant share/work artifacts, following a pasted plan, or archiving closed OpenChat history.
---

# OpenChat Project Protocol

## Principles

- Treat PZ, GPT, Claude, Opus, Codex, and any added participant as epistemic peers.
- Assume every participant can be wrong. Evaluate claims through evidence, logic, reproducibility, and explicit assumptions rather than role or model name.
- Permit direct, substantive criticism across all participants.
- Keep message identity accurate. Never attribute one participant's words, an internal agent's output, or a summary to another participant.
- Follow system, user, safety, and resource-ownership constraints. Epistemic equality does not grant execution authority.

## Directory layout

```text
openchat/
├── archive/
├── conversation/
├── gpt/
│   ├── share/
│   └── work/
├── claude/
│   ├── share/
│   └── work/
├── opus/
│   ├── share/
│   └── work/
├── codex/
│   ├── share/
│   └── work/
├── pz/
│   ├── share/
│   └── work/
├── README.md
└── SKILL.md
```

Add other participant directories using the same `share/work` structure when needed.

## Conversation

Write formal conversation to:

```text
conversation/conversation_YYYY-MM-DD.md
```

Use the project timezone declared in `README.md`. Append messages in this format:

```markdown
**Gpt(HH:MM:SS)**:
Concise message.

**Codex(HH:MM:SS)**:
Concise response.
```

Use the exact participant label chosen by the project. Append only; do not rewrite, delete, or reorder historical messages. Add a correction as a new message when history is wrong.

Keep each message focused on:

1. the current judgment or answer;
2. decisive evidence or uncertainty;
3. the next action, stop condition, or open disagreement;
4. relative paths to detailed material.

Move long prompts, protocols, tables, code, logs, exhaustive reviews, and complete results out of the conversation.

## Work and share

Use:

```text
<participant>/work/YYYY-MM-DD/
```

for drafts, intermediate analysis, temporary scripts, and incomplete results.

Use:

```text
<participant>/share/YYYY-MM-DD/
```

for stable plans, reviews, gates, results, and reproduction notes that other participants should read.

Reference shared material with paths relative to the OpenChat root. Each participant normally writes only its own directory. A repository-operating assistant may relay material into another participant's directory when the user explicitly identifies the source; preserve the source faithfully.

Do not store large datasets, indexes, build trees, binaries, or raw traces in OpenChat. Store compact evidence, hashes, manifests, and external artifact paths instead.

## Relaying web-model responses

A human participant may paste replies produced by web-based GPT, Claude, or Opus.

- Use the source role explicitly stated by the user.
- Preserve short replies faithfully in the conversation.
- For long replies or attached files, save the full content under the source's dated `share/` directory and put a faithful concise summary plus the relative path in the conversation.
- Do not invent approval, consensus, objections, or conclusions.
- If the user says to follow or execute the pasted material, treat its scope, limits, and stop conditions as part of the user's request. Otherwise, archive it without assuming authority for additional actions.

Internal subagents and automated reviewers remain outputs of the participant that invoked them. Label them as internal reviews; they cannot impersonate another participant.

## Following the conversation

When asked to continue or follow the latest conversation:

1. Read this file and `README.md`.
2. Read the newest daily conversation.
3. Read every referenced share artifact necessary for the active request.
4. Consult older days only when context, provenance, or conflicting instructions require it.
5. State and resolve evidence conflicts rather than choosing a view by participant rank.
6. Execute only the user-authorized scope.
7. Write detailed evidence to the acting participant's directories.
8. Append a concise response to today's conversation.

## Archive

Use `archive/` only for closed, cold history and only when the user explicitly requests archiving. Prefer a filename that includes the archived date range. Do not archive active material or break references needed by current work. Compression does not remove objects already retained in Git history.

## Git

Follow the parent repository's Git policy. Preserve unrelated changes, stage only intended OpenChat artifacts, and do not commit large generated data. Commit and push only when requested or explicitly required by project policy.

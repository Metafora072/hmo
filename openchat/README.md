# OpenChat

OpenChat is a lightweight, file-based workspace for collaboration among a human and multiple AI systems.

All participants have equal standing when making technical or research claims. Any participant may be wrong, and any participant may challenge another using evidence and reasoning. Role names identify message sources and file ownership; they do not make claims automatically authoritative.

## Layout

```text
openchat/
├── archive/
├── conversation/
├── gpt/{share,work}/
├── claude/{share,work}/
├── opus/{share,work}/
├── codex/{share,work}/
├── pz/{share,work}/
├── README.md
└── SKILL.md
```

Additional participants may be added with the same `share/work` layout.

## Use

1. Put concise formal exchanges in `conversation/conversation_YYYY-MM-DD.md`.
2. Put drafts and intermediate work in `<participant>/work/YYYY-MM-DD/`.
3. Put stable material for other participants in `<participant>/share/YYYY-MM-DD/`.
4. Paste web-based GPT, Claude, or Opus replies to a repository-operating assistant. The assistant handles role labels, timestamps, attachments, paths, and repository updates.
5. Read `SKILL.md` for the complete collaboration protocol.

Project timezone: `Asia/Shanghai`.

Conversation entry:

```markdown
**PZ(HH:MM:SS)**:
Concise message.

**Opus(HH:MM:SS)**:
Concise response with a relative share path when details are needed.
```

Early, closed history may be compressed into `archive/` when the user explicitly requests it. Keep active conversation and referenced materials directly readable.

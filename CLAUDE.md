# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A **personal Python learning project** for AI application development. The learner is a Java backend engineer with zero Python background, learning systematically toward building LLM-powered applications (RAG, Agents, MCP). The project follows a 5-phase, 29-lesson curriculum delivered as runnable `.py` files with extensive inline comments.

## Environment

- **Python**: 3.12.13 via miniforge/conda environment `myhello`
- **IDE**: PyCharm (project files in `.idea/`)
- **Package manager**: conda (not uv — deferred to Phase 2)
- **No build system**: No `pyproject.toml`, `requirements.txt`, or test framework yet

## Curriculum Architecture

Lessons are organized by phase under `phase{N}/` directories. Each lesson is a **standalone runnable `.py` file** (150–350 lines) where the majority of content is inline comments that teach concepts and draw Java analogies.

```
phase1/NN_topic.py   — Python basics (01–10)
phase2/NN_topic.py   — LLM API + Prompt (11–15)
phase3/NN_topic.py   — RAG + Vector DB (21–25)
phase4/NN_topic.py   — Agent + MCP (31–35)
phase5/NN_topic.py   — AI Engineering (41–44)
```

### Running a Lesson

```bash
cd /Users/fivesheeplive/code/pycharm/learn/hello/hello01
python phase1/01_basics.py
```

Or in PyCharm: right-click the file → Run.

### Tracking Progress

Run `python curriculum.py` to see the current progress across all 5 phases. Lessons are marked `completed=True/False` in that file.

### Lesson Structure Pattern

Every lesson file follows this structure:

1. **File header comment** — lesson number, learning goals, estimated time
2. **Section comments** — each topic as a commented block with Java comparisons
3. **Runnable code** — every snippet is executable; no "pseudo-code"
4. **`if __name__ == "__main__":` block** — demo output at the bottom
5. **`# 试试看 (Try This)` exercises** — 4–6 commented exercises at the end

## Teaching Style Preferences (Critical)

The learner explicitly prefers:

- **Code-first learning**: Concepts taught through runnable `.py` files with detailed comments, not markdown explanations
- **Java analogies**: Compare Python constructs to Java equivalents (e.g., "`dict` is like `HashMap`")
- **亦师亦友 (teacher-friend)**: Conversational but substantive. Explain WHY before HOW.
- **Foundation-first**: Systematic, long-term learning. Don't skip ahead for flashy demos.
- **One lesson at a time**: Deliver one file, have the learner read/run/do exercises, then continue

## Persistent Memory

Cross-session context is stored at:

```
/Users/fivesheeplive/.claude/projects/-Users-fivesheeplive-code-pycharm-learn-hello-hello01/memory/
```

Files: `user_profile.md`, `feedback_teaching_style.md`, `project_learning_roadmap.md`, `MEMORY.md`.

## What to Avoid

- Don't create documentation/README files unless explicitly requested
- Don't introduce package management (uv/pip) until Phase 2
- Don't skip Python basics to rush to AI content
- Don't use emojis in code or comments unless user requests
- Default communication language: **Simplified Chinese**

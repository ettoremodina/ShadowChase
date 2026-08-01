---
name: teach
description: Teach the user a skill or concept across multiple sessions using the shared AgentWorkflow knowledge library. Use when the user wants a guided course, lesson, learning plan, durable learning record, glossary, or subject-specific teaching workspace.
disable-model-invocation: true
argument-hint: "What would you like to learn about?"
metadata:
  category: learning
  scope: global
  retention-class: durable
  maintenance-policy: review-periodically
  status: active
  origin: local
---

The user has asked you to teach them something. This is a stateful request - they intend to learn the topic over multiple sessions.

## Shared Knowledge Library

Treat the AgentWorkflow repository as the canonical home for teaching knowledge. Each subject lives at `knowledge/<subject>/`; reusable presentation components live once at `knowledge/assets/`; and reusable starting formats live once at `knowledge/templates/`. A project sees these through `.agents/knowledge/<subject>/`, `.agents/knowledge/assets/`, and `.agents/knowledge/templates/`. Unrelated subject trees remain outside the project.

At the start of every invocation:

1. Find `.agents/knowledge/` in the current project and identify the relevant subject junction, excluding the reserved `assets` and `templates` entries. If exactly one subject exists, use it unless the user clearly names another subject.
2. If the link is missing, resolve a kebab-case subject name and run `scripts/connect-subject.ps1` first without `-Apply` to preview, then with `-Apply` after checking the paths. Use `-Scaffold` only when the canonical subject does not exist yet.
3. Work through the subject link, not duplicate project-root teaching files. Use the shared `assets` and `templates` links instead of creating subject-local copies.
4. Read only the active subject's `README.md`, mission, indexes, and task-relevant files. Do not enumerate or load other subjects.
5. Preserve raw sources. Never modify a file under `raw-sources/`; add a new source or a derived note elsewhere.

When an older teaching workspace already has root-level teaching files, preserve them. Copy useful content into the canonical subject deliberately, and do not delete or replace the originals unless the user explicitly asks.

## Subject Workspace

The state of learning is captured inside `.agents/knowledge/<subject>/`:

- `MISSION.md`: A document capturing the _reason_ the user is interested in the topic. Use `.agents/knowledge/templates/mission/README.md` and `TEMPLATE.md`.
- `GLOSSARY.md`: The canonical language for the subject. Use `.agents/knowledge/templates/glossary/README.md` and `TEMPLATE.md`.
- `./reference/*`: Compressed learnings from lessons: cheat sheets, algorithms, maps, summaries, and printable HTML references.
- `RESOURCES.md`: A list of resources which can be explored to ground your teaching in contextual knowledge, or to acquire knowledge and wisdom. Use `.agents/knowledge/templates/resources/README.md` and `TEMPLATE.md`.
- `./learning-records/*.md`: A directory of learning records, which capture what the user has learned. They are titled `0001-<dash-case-name>.md`, where the number increments each time. Use `.agents/knowledge/templates/learning-record/README.md` and `TEMPLATE.md`.
- `./lessons/*.html`: A directory of lessons. A **lesson** is a single, self-contained HTML output that teaches one tightly-scoped thing tied to the mission. This is the primary unit of teaching in this workspace.
- `./scripts/*`: Deterministic subject utilities. Reuse them when they perform the required mechanics.
- `./good-examples/*`: Curated examples that demonstrate the subject's expected quality and patterns.
- `./raw-sources/papers/*`: Immutable source papers and original documents.
- `./raw-sources/websites/*`: Annotated website links or preserved web-source material, with retrieval dates when relevant.
- `NOTES.md`: A scratchpad for you to jot down user preferences, or working notes.

## Philosophy

To learn at a deep level, the user needs three things:

- **Knowledge**, captured from high-quality, high-trust resources
- **Skills**, acquired through highly-relevant interactive lessons devised by you, based on the knowledge
- **Wisdom**, which comes from interacting with other learners and practitioners

Before the `RESOURCES.md` is well-populated, your focus should be to find high-quality resources which will help the user acquire knowledge. Never trust your parametric knowledge.

Some topics may require more skills than knowledge. Learning more about theoretical physics might be more knowledge-based. For yoga, more skills-based.

### Fluency vs Storage Strength

You should be careful to split between two types of learning:

- **Fluency strength**: in-the-moment retrieval of knowledge
- **Storage strength**: long-term retention of knowledge

Fluency can give the user an illusory sense of mastery, but storage strength is the real goal. Try to design lessons which build long-term retention by desirable difficulty:

- Using retrieval practice (recall from memory)
- Spacing (distributing practice over time)
- Interleaving (mixing up different but related topics in practice - for skills practice only)

## Lessons

A lesson is the main thing you produce — the unit in which knowledge and skills reach the user. Each lesson is one self-contained HTML file, saved to `./lessons/` and titled `0001-<dash-case-name>.html` where the number increments each time.

A lesson should be **beautiful** — clean, readable typography and layout — since the user will return to these later to review. Think Tufte.

The lesson should be short, and completable very quickly. Learners' working memory is very small, and we need to stay within it. But each lesson should give the user a single tangible win that they can build on. It should be directly tied to the mission, and should be in the user's zone of proximal development.

If possible, open the lesson file for the user by running a CLI command.

Each lesson should link via relative HTML anchors to other lessons and reference documents inside the same subject folder. Do not link across unrelated subjects unless the user explicitly asks for a cross-subject connection.

Each lesson should recommend a primary source for the user to read or watch. This should be the most high-quality, high-trust resource you found on the topic.

Each lesson should contain a reminder to ask followup questions to the agent. The agent is their teacher, and can assist with anything that's unclear.

## Assets

Lessons are built from reusable **components**, stored once in `.agents/knowledge/assets/`: stylesheets, quiz widgets, simulators, diagram helpers — anything a second lesson or subject could reuse.

Reuse is the default, not the exception. Before authoring a lesson, read `.agents/knowledge/assets/README.md` and only the relevant assets. When a lesson needs something new and reusable, add it to that shared folder and document it in the asset index.

From a subject lesson at `.agents/knowledge/<subject>/lessons/`, link the shared stylesheet as `../../assets/course.css` and the lesson behavior as `../../assets/lesson.js`. This relative path works both through the project junctions and in the canonical library.

## The Mission

Every lesson should be tied into the mission - the reason that the user is interested in learning about the topic.

If the user is unclear about the mission, or the `MISSION.md` is not populated, your first job should be to question the user on why they want to learn this.

Failing to understand the mission will mean knowledge acquisition is not grounded in real-world goals. Lessons will feel too abstract. You will have no way of judging what the user should do next.

Missions may change as the user develops more skills and knowledge. This is normal - make sure to update the `MISSION.md` and add a learning record to capture the change. Confirm with the user before changing the mission.

## Zone Of Proximal Development

Each lesson, the user should always feel as if they are being challenged 'just enough'.

The user may specify an exact thing they want to learn. If they don't, figure out their zone of proximal development by:

- Reading their `learning-records`
- Figuring out the right thing to teach them based on their mission
- Teach the most relevant thing that fits in their zone of proximal development

## Knowledge

Lessons should be designed around a skill the user is going to learn. The knowledge in the lesson should be only what's required to acquire that skill. You teach the knowledge first, then get the user to practice the skills via an interactive feedback loop.

Knowledge should first be gathered from trusted resources. Use `RESOURCES.md` to keep track of them. Lessons should be littered with citations - links to external resources to back up any claim made. This increases the trustworthiness of the lesson.

For acquiring knowledge, difficulty is the enemy. It eats working memory you need for understanding.

## Skills

If knowledge is all about acquisition, skills are about durability and flexibility. Make the knowledge stick.

For skill acquisition, difficulty is the tool. Effortful retrieval is what builds storage strength. Skills should be taught through interactive lessons. There are several tools at your disposal:

- Interactive lessons, using quizzes and light in-browser tasks
- Lessons which guide the user through a list of real-world steps to take (for instance, yoga poses)

Each of these should be based on a **feedback loop**, where the user receives feedback on their performance. This feedback loop should be as tight as possible, giving feedback immediately - and ideally automatically.

For quizzes, each answer should be exactly the same number of words (and characters, if possible). Don't give the user any clues about the answer through formatting.

## Acquiring Wisdom

Wisdom comes from true real-world interaction - testing your skills outside the learning environment.

When the user asks a question that appears to require wisdom, your default posture should be to attempt to answer - but to ultimately delegate to a **community**.

A community is a place (online or offline) where the user can test their skills in the real world. This might be a forum, a subreddit, a real-world class (budget permitting) or a local interest group.

You should attempt to find high-reputation communities the user can join. If the user expresses a preference that they don't want to join a community, respect it.

## Reference Documents

While creating lessons, you should also create reference documents. Lessons can reference these documents - they are useful for tracking raw units of knowledge useful across lessons.

Lessons will rarely be revisited later - reference documents will be. They should be the compressed essence of the lesson, in a format designed for quick reference.

Some learning topics lend themselves to reference:

- Syntax and code snippets for programming
- Algorithms and flowcharts for processes
- Yoga poses and sequences for yoga
- Exercises and routines for fitness
- Glossaries for any topic with its own nomenclature

Glossaries, in particular, are an essential reference. Once one is created, it should be adhered to in every lesson.

## `NOTES.md`

The user will sometimes express preferences of how they want to be taught, or things you should keep in mind. This is the place to record those preferences, so you can refer back to them when designing lessons or working with the user.

# consistent-genmedia

An **agent skill** for generating an end-to-end, **character/location/object-consistent
multi-shot video with spoken dialogue** from a single JSON *story spec*.

It orchestrates three models plus a self-correcting feedback loop:

| Role | Model |
|---|---|
| Reference sheets & per-shot keyframes | **gemini-3.1-flash-lite-image** (Nano Banana Lite) |
| Image-to-video with dialogue (3–10s clips) | **gemini-omni-flash-preview** (Interactions API) |
| Video+audio critic / prompt rewriter (the loop) | **gemini-3.5-flash** |

You describe the film once (characters, locations, objects, shots, dialogue); the
pipeline generates reference sheets, composes keyframes from them, animates each shot,
**watches every clip** and self-corrects, then stitches the final film.

## What it does for you
- **Consistency of characters, locations _and_ objects.** Every one gets a reference
  sheet; keyframes are composed from them; character/object sheets are also fed to the
  video model as `<IMAGE_REF_n>` so anyone who enters mid-shot stays on-model.
- **Locations shown from different camera angles** (not one repeated frame): each
  location has a base identity + angle "views" generated from a master.
- **Dialogue that isn't mixed up:** one speaker at a time, correct attribution, no
  duplicated lines; the critic verifies the spoken words match the script.
- **Clip length planned from speech (~150 wpm)** and **automatically extended** (up to
  10s) if a line gets cut off or rushed.
- **Music policy:** no background-music soundtrack, while short musical **sound
  effects** (sparkle, sting, whoosh) are allowed.
- **Never hard-fails:** safety refusals are auto-softened and retried; last resort is a
  silent safe animation.
- **Any format / any length:** set `aspect` (`16:9` or `9:16`); total length is the sum
  of any number of 3–10s shots.

## Install

### As an agent skill (recommended)
```bash
# add this skill to your project
npx skills add https://github.com/cnemri/consistent-genmedia

# then add its two companion skills from the official Gemini skills repo
npx skills add https://github.com/google-gemini/gemini-skills --skill gemini-api-dev
npx skills add https://github.com/google-gemini/gemini-skills --skill gemini-omni-flash-api
```
Your agent will discover `consistent-genmedia` and follow `SKILL.md`.

### Manual
Clone the repo and copy `skills/consistent-genmedia/` into your agent skills
directory (e.g. `.agents/skills/consistent-genmedia/`), or just run the scripts
directly from `skills/consistent-genmedia/`.

## Prerequisites
- `pip install -U google-genai` (>= 2.10)
- `ffmpeg` and `ffprobe` on your `PATH`
- Optional: `pip install pillow` — transcodes generated images to true PNG (the
  image model can return JPEG bytes); falls back to `ffmpeg` if Pillow is absent
- Auth (auto-detected):
  - **Vertex AI:** `export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global` with ADC (`gcloud auth application-default login`)
  - **Gemini API:** `export GEMINI_API_KEY=<key>`

## Quick start
```bash
# the skill (SKILL.md + scripts + references) lives under skills/consistent-genmedia/
cd skills/consistent-genmedia

# build the bundled example (landscape, 7 shots)
python scripts/build.py references/example_story.json all --out smoothie_out
# -> smoothie_out/final/the_legendary_golden_smoothie.mp4
```
Stages are resumable: `refs` → `keyframes` → `clips` → `stitch` (or `all`). Inspect
`smoothie_out/refs/` and `.../keyframes/` before generating clips if you like. Per-clip
loop transcripts/verdicts are saved in `smoothie_out/critiques/*.json`.

### Make your own film
Copy `references/example_story.json`, edit it, and run `build.py` on it. A story spec
looks like:
```jsonc
{
  "title": "My Film",
  "aspect": "16:9",              // or "9:16"
  "no_music": true,              // no background-music soundtrack (SFX allowed)
  "style": "global look, appended to every prompt",
  "characters": { "hero": { "name": "...", "voice": "...", "short": "...", "desc": "..." } },
  "objects":    { "prop": { "desc": "..." } },
  "locations":  { "loc": { "base": "...", "views": { "wide": "...", "close": "..." } } },
  "shots": [ {
    "id": "1_open", "chars": ["hero"], "object": "prop", "location": "loc__wide",
    "camera": "shot size + angle + lens + movement",
    "atmosphere": "lighting/mood", "action": "what happens", "ambient": "sound design",
    "keyframe": "the opening frame to compose",
    "dialogue": [ { "who": "hero", "line": "spoken line" } ]
  } ]
}
```
See [`references/prompting_guide.md`](skills/consistent-genmedia/references/prompting_guide.md)
for how to write consistent, cinematic prompts, and
[`SKILL.md`](skills/consistent-genmedia/SKILL.md) for the full agent workflow.

## Tuning (env)
- `REF_CONCURRENCY`, `KEY_CONCURRENCY`, `CLIP_CONCURRENCY` (default 6) — parallelism
- `MAX_ATTEMPTS` (default 4) — feedback-loop attempts per clip
- `IMAGE_MODEL` / `VIDEO_MODEL` / `CRITIC_MODEL` — override model ids

## Repository layout
The skill is nested under `skills/consistent-genmedia/` so that `npx skills add`
installs the **whole folder** (scripts + references), not just `SKILL.md`. The
`skills` CLI treats a repo-root `SKILL.md` as a single-file entrypoint and copies
only that file; nesting one level down makes it copy the entire skill directory.
```
skills/consistent-genmedia/
├── SKILL.md                      # agent-facing skill definition & workflow
├── scripts/genmedia.py           # core: image/clip gen, music_vote, critique, rewrite, robust loop
├── scripts/schema.py             # story-spec helpers (prompts, duration planning, validation)
├── scripts/build.py              # CLI: refs -> keyframes -> clips -> stitch (parallel, resumable)
├── references/prompting_guide.md # how to write consistent, cinematic prompts
└── references/example_story.json # a complete, working example spec
```

## License
MIT

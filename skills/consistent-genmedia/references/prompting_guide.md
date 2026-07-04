# Prompting guide — consistent-genmedia

How to write the fields of a story spec so the two models produce a *consistent*,
cinematic, dialogue-driven film. This uses **omni's** referencing (`<FIRST_FRAME>`,
`<IMAGE_REF_n>`) — not any `@`-style syntax from other tools.

## The two models & the referencing model
- **gemini-3.1-flash-lite-image** (Nano Banana Lite): makes the reference sheets and
  the per-shot keyframes. It accepts multiple **reference images** to hold identity.
- **gemini-omni-flash-preview**: animates a keyframe into a 3–10s clip with dialogue.
  - `<FIRST_FRAME>` — the first image you pass becomes the opening frame.
  - `<IMAGE_REF_0>`, `<IMAGE_REF_1>`, … — every *additional* image is an identity
    reference (0-indexed, counted separately from the first frame). Bind each in the
    text: `"<IMAGE_REF_0> is Benny (banana head, yellow jacket)"`.
- **gemini-3.5-flash**: watches the finished clip (video **and** audio) and judges it.

## Consistency: characters, locations AND objects
1. **Character bible** — write each character once, in full, and reuse the *exact*
   wording everywhere (`characters[k].desc`). Include a `short` id for `<IMAGE_REF>`
   binding and a fixed `voice` so speech sounds the same across shots.
2. **Reference sheets** — one per character/object (clean studio/product shot). The
   pipeline conditions every keyframe on the relevant sheets.
3. **Feed refs to omni too** — the pipeline passes each shot's character/object
   sheets as `<IMAGE_REF_n>`, so a character who is NOT in the first frame but
   enters mid-shot still looks correct. (This is the fix for "mid-scene entrants
   drift".)

## Locations must vary by camera angle (not one identical frame)
Real coverage of a set shows different angles/parts. So each location has a **base**
identity plus several **views**. The first view is the *master*; the others are
generated **from the master** ("same place, different camera angle"), so they stay
the same room but look different. Map each shot to the view that fits its framing
(e.g. `juicebar__front`, `juicebar__behind`, `juicebar__counter`, `juicebar__corner`).

## Rich cinematography (the `camera` field)
Be specific and filmic. Combine:
- **Shot size**: extreme wide / wide / medium / medium close-up / close-up / macro insert.
- **Angle**: eye-level / low-angle / high-angle / over-the-shoulder / reverse / top-down.
- **Lens**: 28mm wide, 35mm anamorphic, 50mm, 85mm; shallow / deep depth of field.
- **Movement**: static, slow push-in / pull-back, dolly, tracking / Steadicam, crane /
  boom up, tilt-up, pan, gentle orbit, handheld sway, rack focus.
- **Light**: key/fill/rim, neon rim, volumetric god-rays, bloom, practical bulbs.
Example: *"reverse over-the-shoulder from behind the bar, slow push toward the glowing
smoothie with a rack focus from Coco to the drink; 40mm, golden glow blooming into the
lens."*
Put lighting/mood in `atmosphere`, the physical event in `action`.

## Duration is planned from dialogue (~150 wpm) — do NOT default to 10s
`plan_duration` = words / `words_per_sec` (2.5 ≈ 150 wpm) + lead-in + inter-line pauses
+ reaction + optional `pad`, clamped to **3–10s**. Short lines → short clips; wordy
lines → longer clips. If the critic still finds the dialogue **cut off or rushed**, the
feedback loop **increases the clip length (+2s, up to 10s)** and regenerates. Keep a
single shot's dialogue to what fits in ≤10s (~24 words of pure speech); split longer
beats into multiple shots. Total film length = sum of shots (any number).

## Dialogue that doesn't get mixed up
- One speaker at a time. The motion prompt states, for each line, exactly which
  character speaks; every other character keeps their mouth closed and silent.
- Never two characters speaking at once; never the same words twice; only the
  intended lines (no improvised extra speech).
- The critic verifies attribution AND that the spoken words match the script
  (`lines_match_script`); mismatches trigger a rewrite + retry.

## Audio & the music policy
- Default `no_music: true` means **no background-music soundtrack / score / song**.
- **Musical sound effects are allowed** — a sparkle/shimmer chime, a magical sting, a
  tonal whoosh tied to an on-screen action. Describe them in `ambient` as sound effects.
- Always describe the ambient/foley you want; keep dialogue in front.
- Music detection by the critic is stochastic at the ambient/soundtrack boundary, so
  it uses a **majority vote** with a balanced "is there a background soundtrack?" prompt.

## The self-correcting loop (never fails)
generate → (safety block? soften prompt & retry) → critique (video+audio) →
if approved keep it; else: lengthen the clip if dialogue was cut off/rushed, and/or
rewrite the prompt for any other issue, then retry. After the attempt budget it keeps
the best clip; if every attempt is safety-blocked it falls back to a silent safe
animation so a clip always exists.

## Practical tips
- Vary shot sizes across the film (don't shoot everything as a two-shot).
- Reuse identical wording for anything that must stay consistent.
- Keep character count per shot modest; pass their sheets as refs.
- For vertical (`9:16`) just set `aspect` — everything else is identical.
- Reference sheets are generated once and cached; only keyframes/clips regenerate.

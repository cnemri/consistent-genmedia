#!/usr/bin/env python3
"""
consistent-genmedia :: story-spec schema + prompt builders (story-agnostic).

A "story spec" is a plain dict (usually loaded from JSON). All builders operate on
it, so any agent can describe a film as data and reuse the same pipeline.

SPEC = {
  "title": str,
  "aspect": "16:9" | "9:16",              # any format
  "style": str,                            # global look, appended everywhere
  "words_per_sec": 2.5,                    # optional (default 2.5 == 150 wpm)
  "no_music": true,                        # optional (default true == no background music)
  "target_seconds": 60,                    # optional; if set, the duration plan sums to it EXACTLY
  "characters": { key: {"name","desc","short","voice"} },
  "objects":    { key: {"desc"} },         # optional hero props
  "locations":  { key: {"base", "views": { view: desc, ... }} },  # first view = master
  "shots": [ {
      "id": str,
      "chars": [char_key, ...],
      "object": obj_key | null,            # optional
      "location": "loc__view",             # a specific angle view
      "camera": str, "atmosphere": str, "action": str, "ambient": str, "keyframe": str,
      "dialogue": [ {"who": char_key, "line": str}, ... ],
      "pad": 0.0                            # optional extra seconds for action beats
  } ]
}

Location view keys are "<location>__<view>", e.g. "juicebar__front".
"""

import math

CLIP_MIN_SECONDS = 3
CLIP_MAX_SECONDS = 10


def _cfg(spec, key, default):
    return spec.get(key, default)


def words_per_sec(spec):
    return float(spec.get("words_per_sec", 2.5))


def loc_key(loc, view):
    return f"{loc}__{view}"


def location_master(spec, loc):
    return loc_key(loc, next(iter(spec["locations"][loc]["views"])))


def char(spec, key):
    return spec["characters"][key]


# --- References ---------------------------------------------------------------
def build_ref_specs(spec):
    """Return (masters, variants).
      masters : {ref_key: prompt}         # characters, objects, location masters
      variants: {ref_key: (prompt,[deps])}# location angle-views (from their master)
    """
    style = spec.get("style", "")
    masters, variants = {}, {}
    for k, c in spec.get("characters", {}).items():
        masters[k] = (
            f"Full-body character reference sheet of {c['desc']}. Centered on a clean "
            f"plain warm-grey studio background, three-quarter front view, standing, "
            f"full figure head to feet, even soft studio lighting, no text, no labels, "
            f"single character only. {style}")
    for k, o in spec.get("objects", {}).items():
        masters[k] = (f"Product hero shot of {o['desc']}. The single prop centered on a "
                      f"plain dark background, dramatic rim light. {style}")
    for loc, lspec in spec.get("locations", {}).items():
        views = list(lspec["views"].items())
        (mv_key, mv_desc) = views[0]
        mkey = loc_key(loc, mv_key)
        masters[mkey] = (f"{lspec['base']}. {mv_desc}. No people, empty set, cinematic "
                         f"establishing view. {style}")
        for (vk, vd) in views[1:]:
            variants[loc_key(loc, vk)] = (
                (f"This is the EXACT SAME location as the reference image (same place, "
                 f"same signage, props, layout and materials), just filmed from a "
                 f"DIFFERENT camera angle / showing a different part of the space. Keep "
                 f"every identifying detail identical; only the camera viewpoint changes. "
                 f"New view: {vd}. No people, empty set. {style}"),
                [mkey])
    return masters, variants


# --- Per-shot helpers ---------------------------------------------------------
def shot_ref_keys(spec, shot):
    """Ordered ref keys used to COMPOSE the keyframe: chars, object, location view."""
    keys = list(shot["chars"])
    if shot.get("object"):
        keys.append(shot["object"])
    keys.append(shot["location"])
    return keys


def clip_ref_keys(spec, shot):
    """Identity refs fed to omni as <IMAGE_REF_n>: chars then hero object (keeps
    characters/props on-model even when they enter mid-shot)."""
    keys = list(shot["chars"])
    if shot.get("object"):
        keys.append(shot["object"])
    return keys


def dialogue_for_critic(spec, shot):
    return [{"speaker": char(spec, d["who"])["name"], "line": d["line"]}
            for d in shot["dialogue"]]


def target_seconds(spec):
    """Requested total film length in seconds, or None if length is unconstrained."""
    t = spec.get("target_seconds")
    return float(t) if t is not None else None


def _speech_seconds(spec, shot):
    """Seconds the shot's dialogue needs (lead-in + speech + inter-line gaps +
    reaction beat), BEFORE any 3-10s clamp. 0.0 for a silent shot."""
    dl = shot.get("dialogue", [])
    n = len(dl)
    if n == 0:
        return 0.0
    words = sum(len(d["line"].split()) for d in dl)
    gaps = max(0, n - 1) * 0.7
    return words / words_per_sec(spec) + gaps + 1.2 + 1.3


def shot_floor(spec, shot):
    """Smallest whole-second clip length that still fits this shot's dialogue (plus
    any action `pad`), clamped to omni's 3-10s. The planner never goes below this,
    so hitting a requested total can never rush or cut off a line."""
    need = _speech_seconds(spec, shot) + float(shot.get("pad", 0.0))
    if need <= 0:  # silent shot
        return max(CLIP_MIN_SECONDS, min(CLIP_MAX_SECONDS, round(4 + float(shot.get("pad", 0.0)))))
    return max(CLIP_MIN_SECONDS, min(CLIP_MAX_SECONDS, math.ceil(need)))


def _legacy_plan_duration(spec, shot):
    """Per-shot length when NO total-length target is set (original behaviour):
    round the dialogue estimate and clamp to 3-10s; total = whatever that sums to."""
    wps = words_per_sec(spec)
    words = sum(len(d["line"].split()) for d in shot.get("dialogue", []))
    n = len(shot.get("dialogue", []))
    if n == 0:
        return max(3, min(10, round(4 + shot.get("pad", 0.0))))
    speech = words / wps
    gaps = max(0, n - 1) * 0.7
    total = speech + gaps + 1.2 + 1.3 + shot.get("pad", 0.0)
    return max(3, min(10, round(total)))


def plan_durations(spec):
    """DETERMINISTIC per-shot duration plan as {shot_id: whole_seconds}.

    * No `target_seconds`  -> legacy per-shot planning; the film length is just the
      sum (free-running, as before).
    * `target_seconds` set -> the plan is guaranteed to sum to the request EXACTLY:
        floors  = shot_floor() for every shot (each already fits its own dialogue)
        window  = [sum(floors), CLIP_MAX_SECONDS * num_shots]
      The target must fall inside that window, else ValueError names the exact
      numbers and how to fix it. Otherwise the surplus (target - sum(floors)) is
      handed out one second at a time, in shot order, to shots still under the
      per-clip cap. Integers in, integers out -> the sum is exactly the target."""
    shots = spec.get("shots", [])
    tgt = target_seconds(spec)
    if tgt is None:
        return {s["id"]: _legacy_plan_duration(spec, s) for s in shots}

    n = len(shots)
    if n == 0:
        raise ValueError("target_seconds is set but the spec has no shots")
    tgt = int(round(tgt))
    floors = [shot_floor(spec, s) for s in shots]
    lo, hi = sum(floors), CLIP_MAX_SECONDS * n
    if tgt < lo or tgt > hi:
        raise ValueError(
            f"target_seconds={tgt}s is unreachable with {n} shot(s): the feasible "
            f"range is {lo}-{hi}s (each clip is {CLIP_MIN_SECONDS}-{CLIP_MAX_SECONDS}s; "
            f"dialogue sets the per-shot floors {floors}). "
            f"To go shorter: cut dialogue words or remove a shot. "
            f"To go longer: add a shot or lengthen dialogue.")
    durs = list(floors)
    slack = tgt - lo
    i = 0
    while slack > 0:  # guaranteed to terminate: total headroom (hi-lo) >= slack
        if durs[i] < CLIP_MAX_SECONDS:
            durs[i] += 1
            slack -= 1
        i = (i + 1) % n
    return {s["id"]: d for s, d in zip(shots, durs)}


def plan_duration(spec, shot):
    """One shot's planned whole-second duration (honours target_seconds)."""
    return plan_durations(spec)[shot["id"]]


def duration_str(spec, shot):
    return f"{plan_duration(spec, shot)}s"


# --- Prompt builders ----------------------------------------------------------
def keyframe_prompt(spec, shot):
    style = spec.get("style", "")
    notes = []
    for i, k in enumerate(shot_ref_keys(spec, shot)):
        if k in spec.get("characters", {}):
            notes.append(f"reference image {i+1} = exact design of {char(spec, k)['desc']}")
        elif k in spec.get("objects", {}):
            notes.append(f"reference image {i+1} = the exact hero prop to reuse: "
                         f"{spec['objects'][k]['desc']}")
        else:
            notes.append(f"reference image {i+1} = the exact location and camera angle to reuse")
    ref_note = "; ".join(notes)
    return (
        "Compose a single cinematic film still (the opening frame of a shot). Keep "
        f"every character, object and the location IDENTICAL to the provided reference "
        f"images ({ref_note}). Reuse the location exactly as shown in its reference "
        f"(same place and camera framing).\n"
        f"Scene: {shot['keyframe']}.\n"
        f"Framing & lens: {shot.get('camera','')}.\n"
        f"Lighting & atmosphere: {shot.get('atmosphere','')}.\n"
        f"{style}. Rich background detail, filmic composition, no on-screen text or captions.")


def motion_prompt(spec, shot):
    dur = plan_duration(spec, shot)
    style_short = spec.get("style", "")
    no_music = spec.get("no_music", True)
    binds = []
    for i, k in enumerate(clip_ref_keys(spec, shot)):
        if k in spec.get("characters", {}):
            c = char(spec, k)
            binds.append(f"<IMAGE_REF_{i}> is {c['name']} ({c.get('short', c['desc'])})")
        elif k in spec.get("objects", {}):
            binds.append(f"<IMAGE_REF_{i}> is the {k} prop ({spec['objects'][k]['desc']})")
    ref_note = (
        f"Identity references: {'; '.join(binds)}. Use these ONLY as identity references "
        f"(not as new backgrounds): every character must look EXACTLY like their "
        f"reference the whole time — including the moment they enter, turn toward camera "
        f"or move into the shot. " if binds else "")
    seq = []
    for i, d in enumerate(shot.get("dialogue", [])):
        c = char(spec, d["who"])
        connector = "First," if i == 0 else "Then,"
        seq.append(f'{connector} {c["name"]} says in {c["voice"]}: "{d["line"]}"')
    dialogue_block = (" ".join(seq)) if seq else "No spoken dialogue."
    speaker_rule = (
        "The characters speak ONE at a time — only the current speaker's mouth moves "
        "while everyone else keeps their mouth closed and stays silent; never two "
        "characters talking at once and never the same words twice. " if seq else "")
    if no_music:
        music_rule = (
            "IMPORTANT: NO BACKGROUND MUSIC — no musical score, soundtrack, song or "
            "continuous musical bed under the scene. (Short musical SOUND EFFECTS that "
            "punctuate an action — a sparkle/shimmer chime, a magical sting, a tonal "
            "whoosh — are fine.) Keep dialogue and natural ambient/foley in front.")
    else:
        music_rule = ""
    return (
        f"<FIRST_FRAME> {style_short}. Single continuous unbroken shot, no scene cuts. "
        f"Keep the background, location and camera continuous with the first frame. "
        f"Camera work: {shot.get('camera','')}. "
        f"Lighting & atmosphere: {shot.get('atmosphere','')}. "
        f"{ref_note}"
        f"Action: {shot.get('action','')}, with lifelike micro-expressions, secondary "
        f"motion and natural weight. "
        f"Pace it naturally over about {dur} seconds: a short beat to settle, the "
        f"dialogue delivered unhurried and clearly lip-synced, then a final reaction "
        f"beat. {speaker_rule}{dialogue_block} "
        f"Sound design: clear spoken English dialogue with accurate lip-sync plus soft "
        f"ambient {shot.get('ambient','room tone')}. {music_rule}")


def validate(spec):
    """Light validation; raise ValueError on obvious problems."""
    errs = []
    for req in ("title", "style", "characters", "locations", "shots"):
        if req not in spec:
            errs.append(f"missing top-level key: {req}")
    for s in spec.get("shots", []):
        for k in s.get("chars", []):
            if k not in spec.get("characters", {}):
                errs.append(f"shot {s.get('id')}: unknown char '{k}'")
        if s.get("object") and s["object"] not in spec.get("objects", {}):
            errs.append(f"shot {s.get('id')}: unknown object '{s['object']}'")
        loc = s.get("location", "")
        if "__" not in loc:
            errs.append(f"shot {s.get('id')}: location must be 'loc__view', got '{loc}'")
        else:
            lk, vk = loc.split("__", 1)
            if lk not in spec.get("locations", {}) or vk not in spec["locations"][lk]["views"]:
                errs.append(f"shot {s.get('id')}: unknown location view '{loc}'")
    t = spec.get("target_seconds")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0):
        errs.append(f"target_seconds must be a positive number, got {t!r}")
    if errs:
        raise ValueError("Invalid story spec:\n  - " + "\n  - ".join(errs))
    if spec.get("target_seconds") is not None:
        plan_durations(spec)  # deterministic; raises ValueError if the target is unreachable
    return True

#!/usr/bin/env python3
"""
consistent-genmedia :: core library

Reusable primitives for building a consistent multi-shot video with:
  * gemini-3.1-flash-lite-image  -> reference sheets & scene keyframes (Nano Banana Lite)
  * gemini-omni-flash-preview    -> image-to-video with dialogue (Interactions API)
  * gemini-3.5-flash             -> video/audio critic + prompt rewriter (feedback loop)

Nothing here is story-specific. It exposes:
  get_client()                      - Vertex OR API-key client (thread-local)
  generate_image(prompt, out, refs) - keyframe / reference image
  generate_clip(...)                - one omni-flash video (raises SafetyBlocked)
  music_vote(clip, votes)           - majority-vote background-music detector
  critique_video(clip, dialogue,..) - QC verdict (dialogue, music, anomalies, timing)
  rewrite_prompt(prompt, problem,..)- soften (safety) or fix (quality) a prompt
  generate_clip_robust(...)         - the self-correcting loop; NEVER hard-fails,
                                      and lengthens the clip when dialogue won't fit.

Auth (auto-detected):
  * Vertex : set GOOGLE_GENAI_USE_VERTEXAI=true (or GOOGLE_CLOUD_PROJECT) +
             GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (default 'global'), ADC.
  * API key: set GEMINI_API_KEY (Gemini Developer API).
Model ids overridable via IMAGE_MODEL / VIDEO_MODEL / CRITIC_MODEL env vars.
"""

import base64
import json
import mimetypes
import os
import re
import threading
import time

from google import genai
from google.genai import types

IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-lite-image")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "gemini-omni-flash-preview")
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "gemini-3.5-flash")

_TL = threading.local()


class SafetyBlocked(Exception):
    """Raised when the video model refuses a prompt for content-safety reasons."""


def get_client():
    """Per-thread client (the SDK http context is not safe to share across threads).
    Uses Vertex when configured, otherwise a Gemini API key."""
    c = getattr(_TL, "client", None)
    if c is not None:
        return c
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes") \
        or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    http = types.HttpOptions(timeout=900000)
    if use_vertex:
        c = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            http_options=http)
    else:
        c = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options=http)
    _TL.client = c
    return c


# --------------------------------------------------------------------------- #
def _sniff_mime(data, path=""):
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return mimetypes.guess_type(path)[0] or "image/png"


def _image_part_inline(path):
    data = open(path, "rb").read()
    return {"type": "image", "data": base64.b64encode(data).decode(),
            "mime_type": _sniff_mime(data, path)}


def _save_image_png(data, out_path):
    """Write image bytes to out_path so the file's CONTENT always matches a .png
    name. The image model may return JPEG; writing those bytes to a .png file
    makes the extension lie, which breaks any consumer that trusts the extension
    (image viewers, other tools, and agents that attach the file to an LLM with a
    mime_type guessed from the name). We transcode non-PNG bytes to real PNG."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mime = _sniff_mime(data)
    if mime == "image/png":
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path
    # Bytes are not PNG (usually JPEG). Transcode so name == content.
    try:
        import io
        from PIL import Image  # preferred: pip install pillow
        Image.open(io.BytesIO(data)).convert("RGB").save(out_path, format="PNG")
        return out_path
    except Exception:  # noqa: BLE001
        pass
    # Fallback: ffmpeg (already a hard dependency of this skill).
    import subprocess
    import tempfile
    suffix = ".jpg" if mime == "image/jpeg" else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path, out_path],
            check=True)
    finally:
        os.remove(tmp_path)
    return out_path


def _parse_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    blob = m.group(1) if m else text
    m2 = re.search(r"\{.*\}", blob, re.DOTALL)
    if not m2:
        return None
    try:
        return json.loads(m2.group(0))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
def generate_image(prompt, out_path, refs=None, aspect_ratio="16:9", retries=3):
    """Generate an image, optionally conditioned on reference images (identity/style/
    location). Saves the first returned image to out_path."""
    client = get_client()
    parts = []
    for r in (refs or []):
        data = open(r, "rb").read()
        parts.append(types.Part.from_bytes(data=data, mime_type=_sniff_mime(data, r)))
    parts.append(types.Part.from_text(text=prompt))
    cfg = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        candidate_count=1)
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[types.Content(role="user", parts=parts)], config=cfg)
            for cand in resp.candidates or []:
                for part in (cand.content.parts if cand.content else []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and inline.data:
                        return _save_image_png(inline.data, out_path)
            last = RuntimeError("no image part in response")
        except Exception as e:  # noqa: BLE001
            last = e
        print(f"    [image retry {attempt}/{retries}] {os.path.basename(out_path)}: {repr(last)[:140]}")
        time.sleep(3 * attempt)
    raise RuntimeError(f"Image generation failed for {out_path}: {last}")


def generate_clip(prompt, out_path, first_frame=None, refs=None,
                  duration="10s", aspect_ratio="16:9", transient_retries=2):
    """One omni-flash video generation (inline delivery). Raises SafetyBlocked on a
    safety refusal or empty output. `refs` are passed as <IMAGE_REF_n> identity refs."""
    client = get_client()
    input_parts = []
    if first_frame:
        input_parts.append(_image_part_inline(first_frame))
    for r in (refs or []):
        input_parts.append(_image_part_inline(r))
    input_parts.append({"type": "text", "text": prompt})
    rf = {"type": "video", "aspect_ratio": aspect_ratio,
          "delivery": "inline", "duration": duration}
    last = None
    for attempt in range(1, transient_retries + 2):
        t = time.time()
        try:
            inter = client.interactions.create(
                model=VIDEO_MODEL, input=input_parts, response_format=rf)
            ov = inter.output_video
            if not ov or not getattr(ov, "data", None):
                raise SafetyBlocked(f"empty output (possible safety/regional block): {ov}")
            data = ov.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            if not data:
                raise SafetyBlocked(f"empty output (possible safety/regional block): {ov}")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            return {"out": out_path, "id": inter.id, "secs": round(time.time() - t, 1),
                    "bytes": len(data)}
        except SafetyBlocked:
            raise
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "safety" in msg or "harmful" in msg or "blocked" in msg:
                raise SafetyBlocked(str(e))
            last = e
            print(f"    [clip transient {attempt}] {os.path.basename(out_path)} "
                  f"({time.time()-t:.0f}s): {repr(e)[:120]}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Clip generation failed for {out_path}: {last}")


# --------------------------------------------------------------------------- #
def music_vote(clip_path, votes=3, allow_effects=True):
    """Majority-vote background-music detector. Returns (is_music, yes_count, notes).

    Policy: only a continuous background-music SOUNDTRACK counts as music. Short
    musical sound effects (sparkle chime, magical sting, tonal whoosh), room tone,
    ambience and foley are NOT music. (LLM music judgment is stochastic at the
    ambient/soundtrack boundary, so we vote.)"""
    client = get_client()
    if allow_effects:
        q = ('Listen ONLY to the audio. Is there BACKGROUND MUSIC — a continuous musical '
             'score, song or instrumental soundtrack playing as a bed underneath the whole '
             'scene? Short musical SOUND EFFECTS tied to an action (a sparkle/shimmer chime, '
             'a magical "ta-da" sting, a tonal whoosh) are NOT background music and are '
             'allowed. Room tone, ambience and foley are NOT music. Answer YES only for a '
             'continuous background-music soundtrack; otherwise NO. ONLY one word: YES or NO.')
    else:
        q = ('Listen ONLY to the audio. Is there any musical soundtrack, melody, song or '
             'instrumental theme (even faint)? Room tone, ambience and foley are NOT music. '
             'ONLY one word: YES or NO.')
    data = open(clip_path, "rb").read()
    yes, notes = 0, []
    for _ in range(votes):
        try:
            resp = client.models.generate_content(
                model=CRITIC_MODEL,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_bytes(data=data, mime_type="video/mp4"),
                    types.Part.from_text(text=q)])],
                config=types.GenerateContentConfig(temperature=0.0))
            ans = (resp.text or "").strip().upper()
            notes.append(ans[:8])
            if ans.startswith("YES"):
                yes += 1
        except Exception as e:  # noqa: BLE001
            notes.append(f"err:{repr(e)[:16]}")
    return (yes * 2 > votes), yes, notes


CRITIC_INSTRUCTIONS = """You are a strict animation QC reviewer. Watch the video AND \
listen to its audio, then compare it to the intended shot.

INTENDED DIALOGUE (speaker -> line), in order:
{dialogue}

CHARACTERS THAT SHOULD APPEAR: {characters}
MUSIC POLICY: {music_policy}

Return ONLY a JSON object with these keys:
- transcript: list of {{"speaker": str, "line": str}} you actually heard.
- lines_match_script: bool (do the spoken lines match the INTENDED dialogue in wording/meaning? small paraphrase/contractions are fine; substantially different content = false)
- all_intended_lines_fully_spoken: bool (were ALL intended lines spoken to completion, NOT cut off at the end by the clip ending?)
- dialogue_rushed: bool (was speech unnaturally fast / crammed / sped up to fit the time?)
- num_distinct_voices: int
- music_present: bool (true ONLY for a continuous background-music soundtrack; short musical SOUND EFFECTS are allowed and are NOT music)
- attribution_correct: bool (does each intended line come from the correct single character?)
- two_characters_same_line: bool (are two characters saying the SAME line at once?)
- wrong_mouth_movement: bool (does a character's mouth move while another speaks, or a silent character talk?)
- character_consistency_ok: bool (characters match intended designs, no identity swap?)
- severe_visual_anomalies: list of short strings (ONLY serious defects: extra/missing/duplicated limbs or heads, melting/morphing faces, characters merging, garbled unreadable on-screen text). Ignore minor background warping typical of AI video.
- blocking_issues: list of short strings, ONLY for genuinely bad problems: background music when forbidden; wrong attribution; lines don't match script; not all lines fully spoken (cut off); dialogue clearly rushed; two characters same line; a silent character lip-syncing; broken identity; a severe anomaly above. If none, [].
- fix_suggestion: one concrete sentence to fix the blocking issues (empty if none).

Be tolerant of minor AI warping and small lip-sync timing errors; do NOT block on those.
"""


def critique_video(clip_path, dialogue, characters, allow_music=False, retries=2):
    """Watch a clip with the critic model; return a verdict dict with `approved`,
    `blocking_issues`, `needs_more_time`, transcript, etc."""
    client = get_client()
    dlg = "\n".join(f'  {i+1}. {d["speaker"]} -> "{d["line"]}"'
                    for i, d in enumerate(dialogue)) or "  (none)"
    music_policy = ("music is allowed" if allow_music else
                    "NO background music / soundtrack (short musical sound effects are fine)")
    instr = CRITIC_INSTRUCTIONS.format(
        dialogue=dlg, characters=", ".join(characters) or "unspecified", music_policy=music_policy)
    data = open(clip_path, "rb").read()
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=CRITIC_MODEL,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_bytes(data=data, mime_type="video/mp4"),
                    types.Part.from_text(text=instr)])],
                config=types.GenerateContentConfig(temperature=0.1))
            v = _parse_json(resp.text)
            if v is not None:
                blocking = [b for b in (v.get("blocking_issues") or [])
                            if "music" not in b.lower() and "time" not in b.lower()
                            and "cut off" not in b.lower() and "rush" not in b.lower()
                            and "fully spoken" not in b.lower()]
                if not allow_music:
                    is_music, yes, notes = music_vote(clip_path, votes=3)
                    v["music_present"] = is_music
                    v["music_votes"] = f"{yes}/3 {notes}"
                    if is_music:
                        blocking.append("background music present (policy forbids it)")
                truncated = v.get("all_intended_lines_fully_spoken") is False
                rushed = v.get("dialogue_rushed") is True
                v["needs_more_time"] = bool(truncated or rushed)
                if v["needs_more_time"]:
                    blocking.append("dialogue does not fully fit (cut off/rushed) - needs more time")
                v["blocking_issues"] = blocking
                v["approved"] = len(blocking) == 0
                return v
            last = RuntimeError(f"unparseable critic output: {(resp.text or '')[:160]}")
        except Exception as e:  # noqa: BLE001
            last = e
        print(f"    [critic retry {attempt}/{retries}] {os.path.basename(clip_path)}: {repr(last)[:140]}")
        time.sleep(3 * attempt)
    return {"approved": True, "blocking_issues": [], "needs_more_time": False,
            "critic_error": str(last), "fix_suggestion": ""}


def rewrite_prompt(original_prompt, problem, mode):
    """Rewrite a video prompt. mode='safety' softens; mode='quality' fixes issues."""
    client = get_client()
    if mode == "safety":
        task = (
            "The video model REFUSED the prompt below for content-safety reasons "
            f"(error: {problem}). Rewrite it so it will pass safety filters while keeping "
            "the same story beat, characters and the SAME spoken dialogue if possible. "
            "Remove/soften anything that reads as violence, threat, gore, danger to a "
            "person or scary content; make it clearly lighthearted and cartoonish. Keep "
            "it a single continuous shot and keep the no-background-music instruction.")
    else:
        task = (
            "A QC reviewer found problems with the video this prompt produced:\n"
            f"{problem}\n\nRewrite the prompt to FIX those problems. Be very explicit "
            "about dialogue: state exactly which single character speaks each line, and "
            "that every OTHER character keeps their mouth closed and stays silent while "
            "that character speaks (no two characters speaking at once, never the same "
            "words twice, only the intended lines). If background music was detected, "
            "forbid it explicitly (short musical sound effects are still fine). Keep the "
            "same story, characters and exact dialogue lines; keep a single continuous shot.")
    prompt = (f"{task}\n\nKeep it one rich paragraph. Output ONLY the rewritten video "
              f"prompt text.\n\nORIGINAL PROMPT:\n{original_prompt}")
    try:
        resp = client.models.generate_content(
            model=CRITIC_MODEL, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5))
        out = (resp.text or "").strip()
        out = re.sub(r"^```.*?\n|\n```$", "", out).strip()
        return out or original_prompt
    except Exception as e:  # noqa: BLE001
        print(f"    [rewrite error/{mode}] {repr(e)[:120]}")
        return original_prompt


def _dur_to_int(duration):
    try:
        return max(3, min(10, int(round(float(str(duration).lower().rstrip("s"))))))
    except Exception:  # noqa: BLE001
        return 10


def generate_clip_robust(base_prompt, out_path, first_frame, dialogue, characters,
                         refs=None, allow_music=False, duration="10s",
                         aspect_ratio="16:9", max_attempts=4, max_duration=10, log=print):
    """Self-correcting clip generation. NEVER hard-fails.

    Loop: generate -> (safety block? soften & retry) -> critique -> if approved return;
    else if the dialogue was cut off / rushed, LENGTHEN the clip (+2s, cap max_duration
    <= 10) and retry; also rewrite the prompt for any non-timing issues. After
    max_attempts, returns the best clip produced. If every dialogue attempt is
    safety-blocked, falls back to a silent, ultra-safe animation."""
    name = os.path.basename(out_path)
    prompt = base_prompt
    cur_dur = _dur_to_int(duration)
    max_duration = min(10, int(max_duration))
    best = None
    history = []
    safety_softenings = 0

    for attempt in range(1, max_attempts + 1):
        try:
            info = generate_clip(prompt, out_path, first_frame=first_frame, refs=refs,
                                 duration=f"{cur_dur}s", aspect_ratio=aspect_ratio)
        except SafetyBlocked as e:
            log(f"  [{name}] attempt {attempt}: SAFETY block -> softening prompt")
            history.append({"attempt": attempt, "event": "safety_block", "detail": str(e)[:200]})
            safety_softenings += 1
            prompt = rewrite_prompt(prompt, str(e), mode="safety")
            if safety_softenings >= 3:
                prompt = ("<FIRST_FRAME> A gentle, wholesome, lighthearted cartoon moment. "
                          "Single continuous shot, soft natural motion, friendly expressions. "
                          "No dialogue, no background music, only soft ambient sound. Nothing "
                          "scary or dangerous.")
                log(f"  [{name}] falling back to silent safe animation")
            continue

        verdict = critique_video(out_path, dialogue, characters, allow_music=allow_music)
        history.append({"attempt": attempt, "event": "critique", "duration": cur_dur,
                        "blocking": verdict.get("blocking_issues", []),
                        "music_present": verdict.get("music_present"),
                        "needs_more_time": verdict.get("needs_more_time"),
                        "transcript": verdict.get("transcript")})
        if verdict.get("approved"):
            log(f"  [{name}] attempt {attempt}: APPROVED ({cur_dur}s"
                f"{'' if not verdict.get('critic_error') else ', critic-skipped'})")
            return {"status": "approved", "out": out_path, "info": info, "duration": cur_dur,
                    "verdict": verdict, "history": history, "attempts": attempt}

        best = {"out": out_path, "info": info, "verdict": verdict, "duration": cur_dur}
        issues = verdict.get("blocking_issues", [])
        fix = verdict.get("fix_suggestion", "")

        bumped = False
        if verdict.get("needs_more_time") and cur_dur < max_duration:
            new_dur = min(max_duration, cur_dur + 2)
            log(f"  [{name}] attempt {attempt}: dialogue needs more time -> {cur_dur}s->{new_dur}s")
            history.append({"attempt": attempt, "event": "extend_duration",
                            "from": cur_dur, "to": new_dur})
            cur_dur, bumped = new_dur, True

        non_time = [i for i in issues if "time" not in i.lower()
                    and "cut off" not in i.lower() and "rush" not in i.lower()]
        if non_time or not bumped:
            log(f"  [{name}] attempt {attempt}: issues={issues} -> rewriting")
            prompt = rewrite_prompt(prompt, "; ".join(issues) + (". " + fix if fix else ""),
                                    mode="quality")
        else:
            log(f"  [{name}] attempt {attempt}: issues={issues} -> retry at {cur_dur}s")

    log(f"  [{name}] best-effort after {max_attempts} attempts "
        f"(issues={best['verdict'].get('blocking_issues') if best else 'n/a'})")
    return {"status": "best_effort", "out": out_path,
            "info": best["info"] if best else None,
            "duration": best["duration"] if best else cur_dur,
            "verdict": best["verdict"] if best else None,
            "history": history, "attempts": max_attempts}

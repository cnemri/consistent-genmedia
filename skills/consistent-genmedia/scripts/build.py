#!/usr/bin/env python3
"""
consistent-genmedia :: end-to-end builder (CLI)

Turns a story spec (JSON) into a finished, consistent film of ANY aspect ratio and
ANY length (any number of 3-10s shots, stitched).

Usage:
  python build.py <spec.json> [stage] [--out DIR]
    stage = plan | refs | keyframes | clips | stitch | all   (default: all)
    (plan = validate the spec + print the deterministic duration plan and the
            feasible length window; no API calls, no output files)

Stages are resumable (existing outputs are skipped). Parallelism via env:
  REF_CONCURRENCY, KEY_CONCURRENCY, CLIP_CONCURRENCY (default 6)
  MAX_ATTEMPTS (clip feedback-loop attempts, default 4)

Auth: see genmedia.py (Vertex via GOOGLE_CLOUD_PROJECT+ADC, or GEMINI_API_KEY).
Requires: google-genai, ffmpeg + ffprobe.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genmedia as G
import schema as S

REF_CONC = int(os.environ.get("REF_CONCURRENCY", "6"))
KEY_CONC = int(os.environ.get("KEY_CONCURRENCY", "6"))
CLIP_CONC = int(os.environ.get("CLIP_CONCURRENCY", "6"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "4"))

_plock = threading.Lock()


def log(m):
    with _plock:
        print(m, flush=True)


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    S.validate(spec)
    return spec


def dirs(out):
    d = {k: os.path.join(out, k) for k in ("refs", "keyframes", "clips", "final", "critiques")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def exists(p, minsize=10000):
    return os.path.exists(p) and os.path.getsize(p) > minsize


# --------------------------------------------------------------------------- #
def stage_plan(spec):
    """Deterministic, no-API length check. `load_spec` already validated the spec
    (and, if `target_seconds` is set, proved the target is reachable). This prints
    the per-shot duration plan and the feasible length window so you can confirm
    timing BEFORE spending anything on image/video generation."""
    shots = spec["shots"]
    plan = S.plan_durations(spec)
    floors = {s["id"]: S.shot_floor(spec, s) for s in shots}
    n = len(shots)
    lo, hi = sum(floors.values()), S.CLIP_MAX_SECONDS * n
    tgt = S.target_seconds(spec)
    log("== plan ==")
    log(f"  {'shot':16s} {'location':18s} {'floor':>5s} {'planned':>7s}  chars")
    for s in shots:
        sid = s["id"]
        log(f"  {sid:16s} {s.get('location',''):18s} {floors[sid]:>5d} {plan[sid]:>7d}"
            f"  {','.join(s.get('chars', []))}")
    total = sum(plan.values())
    log(f"  {'-' * 60}")
    log(f"  shots: {n}   per-clip: {S.CLIP_MIN_SECONDS}-{S.CLIP_MAX_SECONDS}s   "
        f"feasible total: {lo}-{hi}s")
    if tgt is None:
        log(f"  target_seconds: (unset)   plan total: {total}s  (free-running length)")
    else:
        want = int(round(tgt))
        mark = "EXACT" if total == want else f"MISMATCH (want {want}s)"
        log(f"  target_seconds: {want}s   plan total: {total}s  [{mark}]")
    return total


def stage_refs(spec, D):
    masters, variants = S.build_ref_specs(spec)
    log(f"== refs: {len(masters)} masters + {len(variants)} angle-variants ==")

    def gen_master(item):
        k, prompt = item
        out = os.path.join(D["refs"], f"ref_{k}.png")
        if exists(out):
            return k, "skip"
        G.generate_image(prompt, out, aspect_ratio=spec["aspect"])
        return k, "ok"

    with ThreadPoolExecutor(max_workers=REF_CONC) as ex:
        for f in as_completed([ex.submit(gen_master, it) for it in masters.items()]):
            k, st = f.result(); log(f"  [{st}] master {k}")

    def gen_variant(item):
        k, (prompt, deps) = item
        out = os.path.join(D["refs"], f"ref_{k}.png")
        if exists(out):
            return k, "skip"
        dep_paths = [os.path.join(D["refs"], f"ref_{d}.png") for d in deps]
        G.generate_image(prompt, out, refs=dep_paths, aspect_ratio=spec["aspect"])
        return k, "ok"

    if variants:
        with ThreadPoolExecutor(max_workers=REF_CONC) as ex:
            for f in as_completed([ex.submit(gen_variant, it) for it in variants.items()]):
                k, st = f.result(); log(f"  [{st}] variant {k}")


def stage_keyframes(spec, D):
    log(f"== keyframes: {len(spec['shots'])} ==")

    def one(shot):
        out = os.path.join(D["keyframes"], f"key_{shot['id']}.png")
        if exists(out):
            return shot["id"], "skip"
        refs = [os.path.join(D["refs"], f"ref_{k}.png") for k in S.shot_ref_keys(spec, shot)]
        missing = [r for r in refs if not exists(r)]
        if missing:
            return shot["id"], f"FAIL missing {missing}"
        G.generate_image(S.keyframe_prompt(spec, shot), out, refs=refs, aspect_ratio=spec["aspect"])
        return shot["id"], "ok"

    with ThreadPoolExecutor(max_workers=KEY_CONC) as ex:
        for f in as_completed([ex.submit(one, s) for s in spec["shots"]]):
            sid, st = f.result(); log(f"  [{st}] keyframe {sid}")


def stage_clips(spec, D):
    log(f"== clips (feedback loop, up to {MAX_ATTEMPTS} attempts each) ==")
    allow_music = not spec.get("no_music", True)

    def one(shot):
        out = os.path.join(D["clips"], f"clip_{shot['id']}.mp4")
        if exists(out):
            return {"id": shot["id"], "status": "skip"}
        key = os.path.join(D["keyframes"], f"key_{shot['id']}.png")
        if not exists(key):
            return {"id": shot["id"], "status": "FAIL", "error": "missing keyframe"}
        refs = [os.path.join(D["refs"], f"ref_{k}.png") for k in S.clip_ref_keys(spec, shot)]
        refs = [r for r in refs if exists(r)]
        # With an exact total-length target, don't let the feedback loop grow a clip
        # past its planned length (that would break the length budget); otherwise
        # allow growth up to omni's 10s cap so rushed dialogue can still be fixed.
        max_dur = S.plan_duration(spec, shot) if S.target_seconds(spec) is not None else 10
        res = G.generate_clip_robust(
            base_prompt=S.motion_prompt(spec, shot), out_path=out, first_frame=key,
            dialogue=S.dialogue_for_critic(spec, shot), characters=[spec["characters"][c]["name"] for c in shot["chars"]],
            refs=refs, allow_music=allow_music, duration=S.duration_str(spec, shot),
            aspect_ratio=spec["aspect"], max_attempts=MAX_ATTEMPTS, max_duration=max_dur, log=log)
        with open(os.path.join(D["critiques"], f"{shot['id']}.json"), "w") as fh:
            json.dump({"id": shot["id"], "status": res["status"], "attempts": res["attempts"],
                       "final_duration": res.get("duration"), "verdict": res.get("verdict"),
                       "history": res.get("history")}, fh, indent=2)
        return {"id": shot["id"], "status": res["status"], "attempts": res["attempts"],
                "duration": res.get("duration")}

    results = []
    with ThreadPoolExecutor(max_workers=CLIP_CONC) as ex:
        for f in as_completed([ex.submit(one, s) for s in spec["shots"]]):
            r = f.result(); results.append(r)
            log(f"  => {r['id']}: {r['status']} (attempts={r.get('attempts','-')}, "
                f"{r.get('duration','?')}s){' '+r.get('error','') if r['status']=='FAIL' else ''}")
    return not any(r["status"] == "FAIL" for r in results)


def stage_stitch(spec, D):
    log("== stitch ==")
    clips = [os.path.join(D["clips"], f"clip_{s['id']}.mp4") for s in spec["shots"]]
    missing = [c for c in clips if not exists(c)]
    if missing:
        raise RuntimeError(f"missing clips: {missing}")
    safe = "".join(ch if ch.isalnum() else "_" for ch in spec["title"]).strip("_").lower() or "film"
    out = os.path.join(D["final"], f"{safe}.mp4")
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    n = len(clips)
    # Every clip: de-bed low-frequency rumble + normalise dialogue loudness.
    # When an exact total length is requested, ALSO conform each clip to its planned
    # whole-second duration (pad short clips by holding the last frame + silence,
    # trim long ones) so the stitched film lands on target_seconds deterministically.
    if S.target_seconds(spec) is not None:
        plan = S.plan_durations(spec)
        durs = [plan[s["id"]] for s in spec["shots"]]
        norm = "".join(
            f"[{i}:v]tpad=stop_mode=clone:stop_duration={S.CLIP_MAX_SECONDS},"
            f"trim=0:{t},setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a]highpass=f=85,loudnorm=I=-16:TP=-1.5:LRA=11,"
            f"apad,atrim=0:{t},asetpts=PTS-STARTPTS[a{i}];"
            for i, t in enumerate(durs))
        concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        target_total = sum(durs)
    else:
        norm = "".join(f"[{i}:a]highpass=f=85,loudnorm=I=-16:TP=-1.5:LRA=11[a{i}];" for i in range(n))
        concat_in = "".join(f"[{i}:v][a{i}]" for i in range(n))
        target_total = None
    fc = f"{norm}{concat_in}concat=n={n}:v=1:a=1[v][a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    log(f"  -> {out}  ({dur}s{'' if target_total is None else f', target {target_total}s'})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build a consistent film from a story spec.")
    ap.add_argument("spec", help="path to story spec JSON")
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["plan", "refs", "keyframes", "clips", "stitch", "all"])
    ap.add_argument("--out", help="output directory (default: ./<spec-basename>_out)")
    args = ap.parse_args()

    try:
        spec = load_spec(args.spec)
    except ValueError as e:
        log(f"SPEC INVALID:\n{e}")
        return 2

    if args.stage == "plan":  # deterministic pre-flight: no output dirs, no API
        log(f"### {spec['title']} — {spec['aspect']} — stage: plan ###")
        stage_plan(spec)
        log("Done.")
        return 0

    out = args.out or (os.path.splitext(os.path.basename(args.spec))[0] + "_out")
    D = dirs(out)
    log(f"### {spec['title']} — {spec['aspect']} — stage: {args.stage} — out: {out} ###")

    if args.stage in ("refs", "all"):
        stage_refs(spec, D)
    if args.stage in ("keyframes", "all"):
        stage_keyframes(spec, D)
    if args.stage in ("clips", "all"):
        if not stage_clips(spec, D):
            log("Some clips hard-failed; re-run 'clips' to retry.")
            return 1
    if args.stage in ("stitch", "all"):
        stage_stitch(spec, D)
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

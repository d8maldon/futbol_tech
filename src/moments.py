"""Focused highlight reel: just the key moments (Messi's 3 goals + the disallowed
offside goal), each re-extracted at high fps so it is smooth, the full visual-AI
dashboard rendered over every frame (nothing skipped), then stitched into one clip.

Windows are in SECONDS of the source highlights video (_arg_full.mp4), located by
reading the broadcast scoreboard score/clock; each carries its own match-minute so
the win-probability / xG / ticker panels read correctly for that moment.

    python src/moments.py
"""
import glob
import os
import subprocess

import numpy as np

import match_data as MD
import visual_ai as V

CLIPS = V.CLIPS
FIG = V.FIG
VID = os.path.join(CLIPS, "_arg_full.mp4")
FPS = 20

MOMENTS = [
    dict(key="offside08", label="ALGERIA goal DISALLOWED  -  offside (8')", s=130, d=17, m0=7.5, m1=8.5),
    dict(key="goal17", label="MESSI 1-0 (17')  -  a 0.09 xG strike", s=148, d=24, m0=11.5, m1=17.5),
    dict(key="goal60", label="MESSI 2-0 (60')", s=427, d=23, m0=58.0, m1=60.5),
    dict(key="goal76", label="MESSI 3-0 (76')  -  hat-trick", s=629, d=24, m0=75.0, m1=76.5),
]


def extract(seg):
    d = os.path.join(CLIPS, "mom_" + seg["key"])
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "f_*.jpg")):
        os.remove(f)
    subprocess.run(["ffmpeg", "-y", "-ss", str(seg["s"]), "-i", VID, "-t", str(seg["d"]),
                    "-vf", "fps={}".format(FPS), "-q:v", "3", os.path.join(d, "f_%04d.jpg")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(d, "f_*.jpg")))


def main():
    m = MD.load(V.MATCH_ID)
    parts = []
    for seg in MOMENTS:
        frames = extract(seg)
        cen, team_rgb = V.fit_teams(frames)
        team_rgb = V.display_palette(team_rgb)
        states = V.cv_pass(frames, cen)
        tracked = sum(1 for s in states if s["tracks"])
        print("{}: {} frames, {} with shapes".format(seg["key"], len(states), tracked))
        fmin = np.linspace(seg["m0"], seg["m1"], len(states))
        out = os.path.join(FIG, "_mom_{}.mp4".format(seg["key"]))
        V.render(states, team_rgb, m, out, FPS, "", label=seg["label"], frame_min_override=fmin)
        parts.append(out)

    listfile = os.path.join(FIG, "_concat.txt")
    with open(listfile, "w") as f:
        f.write("\n".join("file '{}'".format(os.path.abspath(p)) for p in parts))
    final = os.path.join(FIG, "wc2026_argentina_goals.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c", "copy", final], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:   # codecs not stream-copyable -> re-encode
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", final], check=True)
    print("wrote", final)


if __name__ == "__main__":
    main()

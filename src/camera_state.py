"""Camera-state classifier: the gate that decides whether a frame can be tracked.

A single broadcast cuts constantly between a WIDE tactical view (trackable), a
TIGHT close-up of a player or two (grass visible but no pitch lock), and OTHER
(graphics / crowd / replay wipes / scoreboards -- no pitch at all). Running the
homography on anything but a wide view produces confident garbage, so we classify
every frame first and gate accordingly:

    wide   -> track normally (homography is reliable)
    tight  -> HOLD the last shape (player on the grass, but can't localise the view)
    other  -> no pitch: blank the top-down, fall back to the data layer (OCR/events)

Cheap, no training: grass-green ratio + edge/text density + whether the keypoint
homography locked + how many players the detector found. Thresholds were tuned by
eye on sampled WC2026 frames (see main()).

    python src/camera_state.py            # validate on the full match, labelled sheet
"""
import glob
import os

import numpy as np

GREEN_WIDE = 0.32      # grass fraction for a wide pitch view
GREEN_TIGHT = 0.20     # grass fraction for a pitch-level close-up
MIN_PLAYERS_WIDE = 5   # a wide tactical view shows several players


def grass_ratio(img):
    """fraction of frame that is pitch-green (HSV gate)"""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h > 32) & (h < 92) & (s > 40) & (v > 40)
    return float(grass.mean())


def edge_density(img):
    """Canny edge fraction -- graphics/scoreboards are edge/text dense"""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float((cv2.Canny(g, 80, 200) > 0).mean())


def classify(img, homography_ok, n_players):
    """return (state, features). state in {wide, tight, other}."""
    green = grass_ratio(img)
    edges = edge_density(img)
    feat = {"green": green, "edges": edges, "h_ok": bool(homography_ok), "n": int(n_players)}
    if homography_ok and green > GREEN_WIDE and n_players >= MIN_PLAYERS_WIDE:
        return "wide", feat
    if green > GREEN_TIGHT and edges < 0.18:        # grass on screen, not a graphic
        return "tight", feat
    return "other", feat


def main():
    import cv2
    from PIL import Image, ImageDraw
    import homography as hg
    from broadcast_track import detect
    ROOT = os.path.join(os.path.dirname(__file__), "..")
    frames = sorted(glob.glob(os.path.join(ROOT, "data", "clips", "argentina_full", "f_*.jpg")))
    sample = frames[::47]
    counts = {"wide": 0, "tight": 0, "other": 0}
    tiles = []
    for fp in sample:
        img = cv2.imread(fp)
        H, _, _ = hg.keypoint_homography(fp)
        players, _, _ = detect(fp)
        state, feat = classify(img, H is not None, len(players))
        counts[state] += 1
        tiles.append((fp, state, feat))
    print("camera-state over {} sampled frames: {}".format(len(sample), counts))
    # labelled contact sheet to eyeball correctness
    cols, tw, th, lab = 6, 200, 112, 16
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + lab)), "#111")
    dr = ImageDraw.Draw(sheet)
    col = {"wide": "#3fb950", "tight": "#ffd23f", "other": "#f0506a"}
    for k, (fp, state, feat) in enumerate(tiles):
        im = Image.open(fp).resize((tw, th))
        cx, cy = (k % cols) * tw, (k // cols) * (th + lab)
        sheet.paste(im, (cx, cy + lab))
        dr.text((cx + 2, cy + 2), "{} g{:.2f} e{:.2f}".format(state, feat["green"], feat["edges"]), fill=col[state])
    out = os.path.join(ROOT, "figures", "_camera_state.png")
    sheet.save(out)
    print("labelled sheet:", out)


if __name__ == "__main__":
    main()

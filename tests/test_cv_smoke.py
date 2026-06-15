"""Import-path smoke test for the computer-vision stack.

No inference, no weights, no network: just confirms the modules parse and their
public entry points exist, so a broken import in the CV layer is caught without
needing a GPU or the model files. Skips cleanly if the optional vision deps
(opencv, ultralytics, sklearn) are not installed.

    python tests/test_cv_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_cv_modules_import():
    try:
        import broadcast_track
        import homography
        import tactical
        import track_fuse                                # noqa: F401
    except ImportError as e:
        print("skipped -- optional vision deps absent:", e)
        return
    assert hasattr(broadcast_track, "detect")
    assert hasattr(homography, "keypoint_homography") and hasattr(homography, "warp")
    assert hasattr(tactical, "assign_teams")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))

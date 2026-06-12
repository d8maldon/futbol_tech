"""Download StatsBomb open-data matches and events for the target tournaments.

Network note: this machine sits behind TLS interception, so certificate
verification is disabled for these read-only public downloads.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3

urllib3.disable_warnings()

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

COMPETITIONS = [
    (43, 3, "WC 2018"),
    (43, 106, "WC 2022"),
    (72, 30, "WWC 2019"),
    (72, 107, "WWC 2023"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
    (55, 282, "Euro 2024"),
    (55, 43, "Euro 2020"),
    (44, 107, "MLS 2023"),
    (1238, 108, "ISL 2021-22"),
]


def fetch(url, path, session):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    r = session.get(url, verify=False, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return "ok"


def main():
    session = requests.Session()
    match_index = []
    for cid, sid, name in COMPETITIONS:
        path = os.path.join(ROOT, "matches", "{}_{}.json".format(cid, sid))
        fetch("{}/matches/{}/{}.json".format(BASE, cid, sid), path, session)
        with open(path, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            match_index.append((m["match_id"], name))
        print("{}: {} matches".format(name, len(matches)))

    print("downloading events for {} matches...".format(len(match_index)))
    done = 0
    failed = []

    def job(mid):
        url = "{}/events/{}.json".format(BASE, mid)
        path = os.path.join(ROOT, "events", "{}.json".format(mid))
        return fetch(url, path, requests.Session())

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(job, mid): mid for mid, _ in match_index}
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append((mid, str(e)))
            done += 1
            if done % 50 == 0:
                print("  {}/{}".format(done, len(match_index)))
                sys.stdout.flush()

    print("done. {} ok, {} failed".format(done - len(failed), len(failed)))
    for mid, err in failed:
        print("FAILED {}: {}".format(mid, err))


if __name__ == "__main__":
    main()

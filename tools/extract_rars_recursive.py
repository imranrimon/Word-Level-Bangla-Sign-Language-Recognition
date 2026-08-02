"""Recursively find and extract every .rar file under a source root.

Used to unpack the BdSLW102_A nested-archive layout:
    data/bdslw102_a_raw/Bangla Sign Language Video Data/Word/*.rar
    -> data/bdslw102_a_videos/Word/<class>/<videos>.mp4

Backend: `rarfile` with bsdtar (from the `libarchive` conda package).
rarfile auto-detects an available extractor; see its BSDTAR_TOOL /
UNRAR_TOOL globals.

Usage:
    python tools/extract_rars_recursive.py \
        --src "data/bdslw102_a_raw/Bangla Sign Language Video Data" \
        --dest data/bdslw102_a_videos \
        --filter Word        # only extract rars whose path contains 'Word'
"""

from __future__ import annotations

import argparse
import os
import sys

import rarfile


def _auto_locate_unrar():
    """Find UnRAR.exe / unrar on Windows even if it's not on PATH."""
    import shutil
    for name in ("unrar", "UnRAR", "UnRAR.exe"):
        p = shutil.which(name)
        if p:
            return p
    # conda-forge installs UnRAR.exe here on Windows but typically not on PATH.
    import sys as _sys
    cands = [
        os.path.join(os.path.dirname(_sys.executable), "..", "Library", "bin", "UnRAR.exe"),
        os.path.join(os.path.dirname(_sys.executable), "Library", "bin", "UnRAR.exe"),
    ]
    for c in cands:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--unrar", default=None,
                    help="full path to UnRAR.exe (auto-detected from conda env if not provided)")
    ap.add_argument(
        "--filter",
        default=None,
        help="substring filter applied to the .rar relative path (case-insensitive)",
    )
    ap.add_argument("--skip-nonempty", action="store_true", default=True)
    ap.add_argument(
        "--no-skip-nonempty", dest="skip_nonempty", action="store_false",
        help="re-extract even if the target directory already has files",
    )
    args = ap.parse_args()

    unrar_path = args.unrar or _auto_locate_unrar()
    if unrar_path:
        rarfile.UNRAR_TOOL = unrar_path
        print(f"using UnRAR: {unrar_path}")
    else:
        print("WARNING: no UnRAR.exe found; rarfile will fall back to bsdtar, "
              "which fails on RAR5 streams. Install via 'conda install -c conda-forge unrar'.")

    os.makedirs(args.dest, exist_ok=True)

    rars = []
    for dirpath, _dirs, files in os.walk(args.src):
        for fn in files:
            if fn.lower().endswith(".rar"):
                path = os.path.join(dirpath, fn)
                if args.filter is None or args.filter.lower() in path.lower():
                    rars.append(path)
    rars.sort()
    print(f"found {len(rars)} .rar files under {args.src}"
          + (f" matching filter={args.filter!r}" if args.filter else ""))

    if not rars:
        return

    failed = []
    for i, rar_path in enumerate(rars, 1):
        rel = os.path.relpath(rar_path, args.src)
        dest = os.path.join(args.dest, os.path.splitext(rel)[0])

        if args.skip_nonempty and os.path.isdir(dest) and os.listdir(dest):
            print(f"[{i}/{len(rars)}] SKIP nonempty: {dest}")
            continue

        os.makedirs(dest, exist_ok=True)
        print(f"[{i}/{len(rars)}] {rel} -> {dest}", flush=True)
        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(dest)
        except Exception as exc:
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
            failed.append(rar_path)

    print(f"done. extracted={len(rars)-len(failed)}  failed={len(failed)}")
    if failed:
        fail_log = os.path.join(args.dest, "_failures.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"failures written to {fail_log}")
        sys.exit(1)


if __name__ == "__main__":
    main()

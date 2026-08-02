"""Extract a zip file with ZIP64 support and progress reporting.

Used to unzip the large (>4 GB, sometimes >40 GB) BdSL dataset archives
that downloaded into `data/`. The MinGW/UnZip combo on Windows can be
flaky on ZIP64; Python's zipfile handles them cleanly.

Usage:
    python tools/extract_zip.py <src.zip> <dest_dir> [--strip-root]
    python tools/extract_zip.py data/BdSLW401.zip data/bdslw401_raw

Set --strip-root to drop the top-level directory in the archive (useful
when a dataset zip nests its contents under a single parent folder).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument(
        "--strip-root",
        action="store_true",
        help="if all entries share a common top-level dir, drop it on extract",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip entries whose destination file already exists",
    )
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"missing zip: {args.src}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.dest, exist_ok=True)

    with zipfile.ZipFile(args.src) as zf:
        members = zf.infolist()
        total_bytes = sum(m.file_size for m in members)
        print(f"src:   {args.src}")
        print(f"dest:  {args.dest}")
        print(f"files: {len(members)}    uncompressed: {total_bytes/1e9:.2f} GB")

        # Detect a common top-level prefix.
        names = [m.filename for m in members if m.filename]
        if args.strip_root and names:
            roots = {n.split("/", 1)[0] for n in names if "/" in n or "\\" in n}
            if len(roots) == 1:
                root = roots.pop() + "/"
                print(f"stripping common root: {root!r}")
            else:
                root = None
                print(f"--strip-root requested but found {len(roots)} top-level dirs; leaving unchanged")
        else:
            root = None

        t0 = time.time()
        bytes_done = 0
        for i, m in enumerate(members):
            target_name = m.filename
            if root and target_name.startswith(root):
                target_name = target_name[len(root):]
                if not target_name:
                    continue
            target = os.path.join(args.dest, target_name)
            if m.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            if args.skip_existing and os.path.exists(target) and os.path.getsize(target) == m.file_size:
                bytes_done += m.file_size
                continue
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with zf.open(m) as src_f, open(target, "wb") as dst_f:
                while True:
                    chunk = src_f.read(1024 * 1024)
                    if not chunk:
                        break
                    dst_f.write(chunk)
            bytes_done += m.file_size

            if (i + 1) % args.report_every == 0 or (i + 1) == len(members):
                pct = 100.0 * bytes_done / max(1, total_bytes)
                elapsed = time.time() - t0
                rate = bytes_done / max(0.1, elapsed) / 1e6
                eta = (total_bytes - bytes_done) / max(1, bytes_done) * elapsed
                print(
                    f"[{i+1}/{len(members)}]  {bytes_done/1e9:.2f}/{total_bytes/1e9:.2f} GB "
                    f"({pct:.1f}%)  {rate:.1f} MB/s  eta {eta/60:.1f} min"
                )

    print("done.")


if __name__ == "__main__":
    main()

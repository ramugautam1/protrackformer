"""Track puncta in a 3D movie.

- input: the movie and a segmentation of it (instance labels, or a binary mask that is split into
  6-connected components of at least 6 voxels)
- output folder:
  - `tracks.txt`: one row per track, `id start end parent_1 ... parent_K` (CTC-DAG track table)
  - `tracked_masks.tif`: T x Z x Y x X, every voxel labelled with its track id
  - `links.csv`: every link of the lineage, `t_src, id_src, t_dst, id_dst`
  - `events.csv`: every split and merge with its frame, time, tracks and position in um
  - `summary.json`: counts, settings and run time
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tifffile

from . import __version__
from .inference import Tracker
from .io import read_stack, to_instances, write_result

log = logging.getLogger("protrackformer")


def voxel_size(path: Path) -> tuple[float, float, float] | None:
    """(z, y, x) in um from ImageJ TIFF metadata, if present."""
    try:
        with tifffile.TiffFile(path) as tf:
            meta = tf.imagej_metadata or {}
            res = tf.pages[0].tags.get("XResolution"), tf.pages[0].tags.get("YResolution")
            if "spacing" not in meta or None in res:
                return None
            x = res[0].value[1] / res[0].value[0]
            y = res[1].value[1] / res[1].value[0]
            return float(meta["spacing"]), float(y), float(x)
    except (OSError, ValueError, ZeroDivisionError, tifffile.TiffFileError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track puncta that split and merge in a 3D movie.")
    ap.add_argument("--image", type=Path, required=True, help="movie: 4D/5D TIFF, NIfTI, or a folder of t###.tif")
    ap.add_argument("--masks", type=Path, required=True, help="segmentation of the movie (labels or binary)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channel", type=int, help="channel of a multi-channel movie")
    ap.add_argument("--frames", help="first:last frame, inclusive, 0-based")
    ap.add_argument("--voxel-size", type=float, nargs=3, metavar=("Z", "Y", "X"), help="um (default: from metadata)")
    ap.add_argument("--frame-interval", type=float, help="seconds between frames")
    ap.add_argument("--cover", action="store_true", help="masks are an annotated foreground: keep every component")
    ap.add_argument("--decode", choices=["auto", "exact", "blocks"], default="auto",
                    help="one integer program, or overlapping blocks for large movies (auto: by size)")
    ap.add_argument("--tile", type=int, default=96, help="tile size for prediction on large fields (0: none)")
    ap.add_argument("--workers", type=int, help="parallel block solves")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    tracker = Tracker(a.checkpoint, a.device)
    frames = tuple(int(v) for v in a.frames.split(":")) if a.frames else None
    image = read_stack(a.image, a.channel, frames)
    masks = read_stack(a.masks, None, frames)
    if tracker.meta.get("masks") != "foreground":
        masks = to_instances(masks)
    if image.shape != masks.shape:
        raise SystemExit(f"image {image.shape} and masks {masks.shape} differ")
    voxel = tuple(a.voxel_size) if a.voxel_size else voxel_size(a.image) or (0.5, 0.124, 0.124)
    log.info("movie %s, voxel %s um, checkpoint %s", image.shape, voxel, a.checkpoint.name)

    start = time.time()
    lineage, labels, g = tracker.track(image, masks, cover=a.cover or None, tile=a.tile or None,
                                       blocks={"auto": None, "exact": False, "blocks": True}[a.decode],
                                       workers=a.workers)
    info = {"image": str(a.image), "masks": str(a.masks), "checkpoint": str(a.checkpoint),
            "decoder": tracker.decoder, "candidate_objects": int(len(g.node_t)),
            "candidate_links": int(len(g.edge_src)),
            "seconds": round(time.time() - start, 1), "version": __version__}
    summary = write_result(a.out, lineage, labels.astype(np.uint32), voxel, a.frame_interval, info)
    log.info(json.dumps({k: summary[k] for k in ("tracks", "objects", "links", "splits", "merges")}))


if __name__ == "__main__":
    main()

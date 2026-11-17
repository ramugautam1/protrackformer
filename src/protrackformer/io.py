"""Reading movies and masks, and writing tracking results.

- movies and masks: a 4D TIFF (T, Z, Y, X), a 5D TIFF with a channel axis, a folder of
  per-frame `t###.tif` / `mask###.tif` files, or a NIfTI file (x, y, z, t)
- ground truth: the CTC-DAG layout (see data/README.md)
- results: `tracks.txt`, `tracked_masks.tif`, `links.csv`, `events.csv`, `summary.json`
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from .lineage import Lineage, presence, read_track_table, write_links, write_track_table

FRAME_PATTERNS = (r"t\d+\.tif", r"man_track\d+\.tif", r"mask\d+\.tif", r"man_seg\d+\.tif", r"man_inst\d+\.tif")


def _frame_files(folder: Path) -> list[Path]:
    for pat in FRAME_PATTERNS:
        files = sorted(f for f in folder.iterdir() if re.fullmatch(pat, f.name))
        if files:
            return files
    raise FileNotFoundError(f"no per-frame TIFF files in {folder}")


def read_stack(path: Path, channel: int | None = None, frames: tuple[int, int] | None = None) -> np.ndarray:
    """Return a T x Z x Y x X array. `frames` = (first, last), inclusive."""
    path = Path(path)
    if path.is_dir():
        files = _frame_files(path)
        if frames:
            files = files[frames[0]:frames[1] + 1]
        return np.stack([tifffile.imread(f) for f in files])
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        arr = np.asarray(nib.load(str(path)).dataobj)
        arr = arr.transpose(3, 2, 1, 0) if arr.ndim == 4 else arr.transpose(2, 1, 0)[None]
    else:
        with tifffile.TiffFile(path) as tf:
            arr, axes = tf.series[0].asarray(), tf.series[0].axes.upper()
        if "C" in axes:
            if channel is None:
                raise ValueError(f"{path.name} has a channel axis ({axes}); pass --channel")
            arr = np.take(arr, channel, axis=axes.index("C"))
            axes = axes.replace("C", "")
        if axes in ("ZYX", "QYX") or arr.ndim == 3:
            arr = arr[None]
        elif arr.ndim != 4:
            raise ValueError(f"cannot read {path.name} with axes {axes} as T, Z, Y, X")
    if frames:
        arr = arr[frames[0]:frames[1] + 1]
    return np.ascontiguousarray(arr)


def to_instances(masks: np.ndarray, connectivity: int = 6, min_size: int = 6) -> np.ndarray:
    """Binary masks become connected components of at least `min_size` voxels; label images pass through.

    A mask is binary when its foreground has a single value: 1, or 255 / 65535 as written by many tools
    (ProTrack3D, ImageJ). A label image has more than one foreground value."""
    fg = masks[masks > 0]
    if fg.size and fg.min() != fg.max():
        return masks.astype(np.int32)
    struct = ndi.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    out = np.zeros(masks.shape, np.int32)
    for t in range(len(masks)):
        lab, _ = ndi.label(masks[t] > 0, struct)
        sizes = np.bincount(lab.ravel())
        small = sizes < min_size
        small[0] = True
        lab[small[lab]] = 0
        out[t] = np.unique(lab, return_inverse=True)[1].reshape(lab.shape)
    return out


def read_sequence(seq_dir: Path) -> tuple[np.ndarray, np.ndarray, Lineage]:
    """A CTC-DAG sequence: images, track-labelled masks and the lineage."""
    seq_dir = Path(seq_dir)
    tra = seq_dir.parent / f"{seq_dir.name}_GT" / "TRA"
    image, labels = read_stack(seq_dir), read_stack(tra)
    table = tra / "man_track_multi.txt"
    lineage = Lineage.from_tracks(read_track_table(table if table.exists() else tra / "man_track.txt"),
                                  presence(labels))
    return image, labels, lineage


def write_result(out: Path, lineage: Lineage, labels: np.ndarray, voxel_um=(0.5, 0.124, 0.124),
                 frame_s: float | None = None, info: dict | None = None) -> dict:
    """Write a tracking result. `lineage` nodes are `(t, label)` of objects in `labels`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    node_tid, table = lineage.to_tracks()
    dtype = np.uint16 if len(table) < 65535 else np.uint32
    tracked = np.zeros(labels.shape, dtype)
    for t in range(len(labels)):
        lut = np.zeros(int(labels[t].max()) + 1, dtype)
        for (tt, lab), tid in node_tid.items():
            if tt == t:
                lut[lab] = tid
        tracked[t] = lut[labels[t]]
    write_track_table(table, out / "tracks.txt")
    write_tracked_masks(out / "tracked_masks.tif", tracked, voxel_um, frame_s)
    tracks = lineage.relabel(node_tid)
    write_links(tracks, out / "links.csv")
    events_table(tracks, tracked, voxel_um, frame_s).to_csv(out / "events.csv", index=False)
    summary = {**tracks.summary(), "tracks": len(table), "frames": len(labels), "voxel_um_zyx": list(voxel_um),
               "frame_interval_s": frame_s, **(info or {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def write_tracked_masks(path: Path, tracked: np.ndarray, voxel_um, frame_s: float | None) -> None:
    """T x Z x Y x X track ids, as an ImageJ hyperstack where possible (opens in Fiji and napari)."""
    if tracked.dtype == np.uint16:
        meta = {"axes": "TZYX", "spacing": voxel_um[0], "unit": "um"}
        if frame_s:
            meta["finterval"] = frame_s
        tifffile.imwrite(path, tracked, imagej=True, metadata=meta,
                         resolution=(1 / voxel_um[2], 1 / voxel_um[1]), compression="zlib")
    else:
        tifffile.imwrite(path, tracked, compression="zlib", metadata={"axes": "TZYX"})


def events_table(tracks: Lineage, tracked: np.ndarray, voxel_um, frame_s: float | None) -> pd.DataFrame:
    """One row per split or merge; the anchor is the parent of a split or the product of a merge."""
    rows = []
    for e in tracks.events():
        t, tid = e.anchor
        z, y, x = ndi.center_of_mass(tracked[t] == tid)
        rows.append({"kind": e.kind, "frame": e.t, "time_s": round(e.t * frame_s, 3) if frame_s else None,
                     "track": tid, "partners": " ".join(str(p[1]) for p in e.partners), "n_partners": len(e.partners),
                     "z_um": round(z * voxel_um[0], 3), "y_um": round(y * voxel_um[1], 3),
                     "x_um": round(x * voxel_um[2], 3)})
    cols = ["kind", "frame", "time_s", "track", "partners", "n_partners", "z_um", "y_um", "x_um"]
    return pd.DataFrame(rows, columns=cols)


def read_result(out: Path) -> tuple[Lineage, np.ndarray]:
    """Read a result written by `write_result` (or a CTC-DAG `TRA` folder)."""
    out = Path(out)
    if (out / "tracks.txt").exists():
        labels = read_stack(out / "tracked_masks.tif")
        return Lineage.from_tracks(read_track_table(out / "tracks.txt"), presence(labels)), labels
    labels = read_stack(out)
    table = next(out / n for n in ("man_track_multi.txt", "res_track.txt", "man_track.txt") if (out / n).exists())
    return Lineage.from_tracks(read_track_table(table), presence(labels)), labels

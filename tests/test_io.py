import numpy as np
import tifffile

from protrackformer.io import read_result, read_stack, to_instances, write_result


def test_result_round_trip(toy, tmp_path):
    image, labels, lin = toy
    summary = write_result(tmp_path, lin, labels, frame_s=7.5)
    assert (summary["splits"], summary["merges"]) == (1, 1)
    for name in ("tracks.txt", "tracked_masks.tif", "links.csv", "events.csv", "summary.json"):
        assert (tmp_path / name).exists()
    back, tracked = read_result(tmp_path)
    assert tracked.shape == labels.shape
    assert len(back.nodes) == len(lin.nodes) and len(back.edges) == len(lin.edges)
    assert sorted(e.kind for e in back.events()) == ["merge", "split"]


def test_read_stack_forms(toy, tmp_path):
    image, _, _ = toy
    tifffile.imwrite(tmp_path / "movie.tif", image, imagej=True, metadata={"axes": "TZYX"})
    assert read_stack(tmp_path / "movie.tif").shape == image.shape
    two = np.stack([image, image // 2], 2)
    tifffile.imwrite(tmp_path / "two.tif", two, imagej=True, metadata={"axes": "TZCYX"})
    assert np.array_equal(read_stack(tmp_path / "two.tif", channel=1), image // 2)
    (tmp_path / "frames").mkdir()
    for t in range(3):
        tifffile.imwrite(tmp_path / "frames" / f"t{t:03d}.tif", image[t])
    assert read_stack(tmp_path / "frames").shape == (3, *image.shape[1:])
    assert read_stack(tmp_path / "movie.tif", frames=(2, 4)).shape[0] == 3


def test_binary_masks_become_components(toy):
    _, labels, _ = toy
    inst = to_instances((labels > 0).astype(np.uint8))
    assert inst.max() >= 2 and inst.dtype == np.int32


def test_binary_masks_with_any_foreground_value(toy):
    """0/255 and 0/65535 masks (ProTrack3D, ImageJ) are binary, not one label covering the whole foreground."""
    _, labels, _ = toy
    ref = to_instances((labels > 0).astype(np.uint8))
    for value, dtype in ((255, np.uint8), (65535, np.uint16)):
        inst = to_instances(((labels > 0) * value).astype(dtype))
        assert np.array_equal(inst, ref)


def test_label_images_pass_through(toy):
    _, labels, _ = toy
    assert np.array_equal(to_instances(labels), labels.astype(np.int32))


# Checkpoints

| file | candidates | decoder | trained on | validation |
|---|---|---|---|---|
| `bazooka_instances.pt` | the given instances | threshold 0.5 | 25 Bazooka crops | Crop03, Crop06, Crop13 |
| `bazooka_segmentation.pt` | segmenter objects and their watershed parts | integer program, λ = 2, µ = 1 | the 18 training crops of fold 0 | its 3 validation crops |
| `gish_segmentation.pt` | segmenter objects and their watershed parts | integer program, λ = 1, µ = 0.5 | Gish quadrants 0-2, from `bazooka_segmentation.pt` | quadrant 3 |

- `bazooka_instances.pt` expects one label per punctum, for example corrected or hand-drawn masks
- the segmentation models expect the output of a segmenter. They were trained on face-connected components of at
  least 6 voxels of an FC-DenseNet segmentation, which is what `protrackformer-track` makes of a binary mask
- distances inside the model are in xy pixels (Bazooka 0.124 µm, Gish 0.132 µm) and z is scaled by 0.5 / 0.124;
  for data with a very different pixel size, train on that data

Each file holds the weights, the network and graph settings, the decoder chosen on validation, and the names of the
training and validation sequences:

```python
import torch

ck = torch.load("checkpoints/gish_segmentation.pt", weights_only=False)
ck["decoder"], ck["train"], ck["description"]
```

## Installation

```bash
# Clone fork
git clone https://github.com/GITHUB_USERNAME/segmentation_inference.git

# Move into the cloned repo folder
cd segmentation_inference

# Create conda environment
conda create -c conda-forge -n segmentation_inference --yes python=3.12
conda activate segmentation_inference

# Install ipykernel for Jupyter notebooks
conda install -c conda-forge ipykernel

# Install package in editable mode
pip install -e .

# OR install package in editable mode to include dev tools
pip install -e ".[dev]"
```


## Usage

The example below demonstrates how to convert an MVBS dataset to a model input tensor, load a trained binary hake segmentation model, and generate softmax scores aligned with the MVBS coordinates.

```python
import torch
import xarray as xr

from segmentation_inference.model import binary_hake_model
from segmentation_inference.utils import get_MVBS_tensor

# Load MVBS dataset
MVBS_ds = xr.open_zarr("path/to/MVBS.zarr")

# Convert MVBS data to model input tensor
MVBS_tensor = get_MVBS_tensor(MVBS_ds, max_depth=590)

# Load trained model checkpoint
model = binary_hake_model().eval()
model.load_state_dict(
    torch.load(
        "path/to/binary_hake_model.ckpt",
        map_location=torch.device("cpu"),
    )["state_dict"]
)

# Run inference
output_dict = model.forward(MVBS_tensor, softmax_temperature=1.0)

# Create xarray DataArray of softmax scores
softmax_score_da = xr.DataArray(
    output_dict["softmax_output"].detach(),
    coords={
        "scatterer_class": ["background", "hake"],
        "depth": MVBS_ds["depth"].sel(depth=slice(None, 590)).values,
        "ping_time": MVBS_ds["ping_time"].values,
    },
    name="softmax_score",
)
```


### Outputs
In particular, the `binary_hake_model` returns a dictionary with the following entries:
- `backbone_output`: Feature maps produced by the backbone network before being passed into the classifier.
- `classifier_output`: Raw classifier values for background and hake before interpolation.
- `interpolated_output`: Classifier values interpolated back to the original MVBS grid.
- `softmax_output`: Softmax values of the interpolated classifier output.

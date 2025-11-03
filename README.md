# Cell Plasmolysis Toolkit

Segment, track, and quantify cell geometry over time to study plasmolysis. The toolkit provides:

* `track.py` — segments cells in time-lapse TIFF stacks, tiles large fields of view, tracks instances across frames, and saves per-tile videos and label stacks.  
* `cell_features.py` — computes per-cell size metrics across time, exports plots, movies, “before/after” images, and a summary CSV. 

---

## Quick start

### Option A — Docker (recommended)

Download existing image (recommended):

```bash
docker pull mdyakova/cell_plasmolysis:v1
```

or build the image (pre-caches Cellpose CPSAM weights during build for faster first run):

```bash
docker build -t cell_plasmolysis:v1 .
```

Run segmentation + tracking (with GPU and a bind mount; adjust paths as needed):

```bash
docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_plasmolysis:v1 python track.py --image_directory "/cell_tracking_files/data/Acetate" --output_directory "/cell_tracking_files/tracking_results" --tile_size 400 --name_filter roi0 --frames_exclude_file "/cell_tracking_files/ROIs to be segmented 20251029.xlsx"
```

Compute features:

```bash
docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_plasmolysis:v1 python cell_features.py --labels_directory "/cell_tracking_files/tracking_results" --output_directory "/cell_tracking_files/final_results" --selected_ids "/cell_tracking_files/Acetate Label Selection.xlsx"
```

### Option B — Local Python (no Docker)

1. Install Python 3.8 and dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run:

```bash
python3.8 main/track.py \
  --image_directory "../Avik/20250624_All_Images_for_Masha/New Turgor Analysis" \
  --output_directory "../Avik/tracking_092225" \
  --tile_size 400

python3.8 main/cell_features.py \
  --labels_directory "../Avik/../Avik/tracking_251016/Acetate" \
  --output_directory "../Avik/results_cells_20251030" \
  --selected_ids "../Avik/20250624_All_Images_for_Masha/Acetate Label Selection.xlsx"
```

Dependencies are pinned in `requirements.txt`. 

---

## What the scripts do

### `track.py` (segmentation + tracking)

* Loads a Cellpose **CPSAM** model on GPU and processes each frame.
* Splits each 3D stack into overlapping XY tiles; runs segmentation at 0°, 90°, 180°, 270° rotations; merges masks that correspond to the same cell; tracks them across frames with **trackpy**; and saves per-tile outputs:

  * `*_x_{j}_y_{i}.tif` (uint16 intensity video)
  * `*_x_{j}_y_{i}_labels.tif` (uint16 label video)
  * `processed_files.txt` (append-only log to skip completed tiles)
    Parameters include `--frames_exclude_file` to drop bad frames per movie and `--name_filter` to restrict which files run.  

### `cell_features.py` (feature extraction)

* For each `*_labels.tif` in `--labels_directory`, pairs it with the raw tile video, rotates each cell to horizontal, rescales (×3) for accurate line drawing, draws a bounding box and a dashed centerline, extracts intensity profiles, and computes:

  * `length_bb`, `width_bb` (bounding-box-based)
  * `length_center` (centerline intensity span above a threshold)
* Outputs:

  * `<output>/<folder>/<subfolder>/movie/<file>*.tif` — a per-cell movie with markers
  * `<output>/<folder>/<subfolder>/individual/<file>.jpg` — “before/after” panel + profiles
  * `<output>/<folder>/<subfolder>/before_after/before.tif` and `after.tif`
  * `<output>/cell_sizes.csv` — summary of all per-frame metrics
    Optional filtering via an Excel sheet of selected labels. 

---

## Input/Output layout (typical)

```
image_directory/
  sampleA_... .tif        # time-lapse z=frames, y, x
output_directory/
  sampleA_x_000_y_000.tif
  sampleA_x_000_y_000_labels.tif
  processed_files.txt
```

Downstream features:

```
features_output/
  <folder>/<subfolder>/
    movie/<folder>_<subfolder>_<cellid>.tif
    individual/<folder>_<subfolder>_<cellid>.jpg
    before_after/before.tif
    before_after/after.tif
  cell_sizes.csv
```

---

## Command-line arguments

### `track.py`

* `--image_directory` (str, required): folder with TIFF stacks (searched recursively).
* `--output_directory` (str, required): where outputs are written.
* `--tile_size` (int, required): tile edge in pixels (overlap of 50 px is applied).
* `--name_filter` (str, default `""`): run only files whose path contains this substring.
* `--frames_exclude_file` (str, optional): Excel file with columns:

  * `Filename` — basename of the movie (e.g., `movie01.tif`)
  * `Frames to exclude` — semicolon-separated list (e.g., `0;1;2`)
    Matching frames are removed before processing. 

### `cell_features.py`

* `--labels_directory` (str, required): folder with `*_labels.tif` files.
* `--output_directory` (str, required): destination for plots/movies/CSV.
* `--k_pixels` (int, default `0`): threshold for normalized edge detection along X/Y
  (used to find box edges from mean profiles; larger → tighter box).
* `--k_intens` (int, default `0`): threshold for normalized centerline intensity
  (used to measure `length_center`; larger → stricter).
* `--selected_ids` (str, optional): Excel file to restrict analysis to specific labels with columns:

  * `Filename` — matches `*_labels.tif` basename (forward-filled per file)
  * `Label` — integer label ID to include
    **Note:** In the code these thresholds are parsed as integers; set `1`…`255` style integers to tune strictness. 

---

## Examples of Excel helpers

**Frames to exclude (for `track.py`):**

| Filename    | Frames to exclude |
| ----------- | ----------------- |
| movie01.tif | 0;1;2             |
| movie02.tif | 5;6               |

**Selected labels (for `cell_features.py`):**

| Filename                   | Label |
| -------------------------- | ----- |
| sampleA_x_000_y_000_labels | 12    |
| sampleA_x_000_y_000_labels | 27    |
| sampleB_x_050_y_050_labels | 3     |

(The `Filename` column is forward-filled by the script when blank.) 

---

## Requirements

* Python 3.8
* CUDA-capable GPU (recommended) + NVIDIA drivers for the Docker `--gpus all` option
* Packages pinned in `requirements.txt` (Cellpose 4.0.4, trackpy 0.7, scikit-image 0.21, OpenCV 4.11, etc.). 

The provided Dockerfile uses `python:3.8-slim`, adds GUI libs for OpenCV wheels, installs requirements, copies `main/` as the working directory, **and pre-caches CPSAM weights** during the image build so first inference runs faster. 

---

## Tips & troubleshooting

* **GPU vs CPU:** The scripts request GPU for Cellpose (`CellposeModel(gpu=True)`). If you must run on CPU, change to `gpu=False` in code or ensure no GPU is visible in the container. 
* **Large frames:** Use a `--tile_size` that fits GPU memory. Tiles overlap by 50 px internally; outputs are per tile. 
* **Skipping finished tiles:** The pipeline appends file names to `processed_files.txt`. Delete entries to re-process tiles. 
* **Bad frames:** Provide the Excel described above to drop problematic frames prior to segmentation. 
* **Feature thresholds:** If your boxes look too tight/loose or centerline spans are off, increase/decrease `--k_pixels` / `--k_intens` (integers). 

---

## Citation

If you use this toolkit in your research, please cite Cellpose and trackpy as appropriate and reference this repository.

---

## License

MIT License

---

## Acknowledgements

* Segmentation powered by **Cellpose CPSAM**.
* Tracking powered by **trackpy**.
* Merging of rotated predictions uses custom contour-merging utilities. 

---

**Repository layout**

```
main/
  track.py
  cell_features.py
  utilities.py
  requirements.txt
  Dockerfile
  README.md  <-- (this file)
```

* `track.py` — core tiling/segmentation/tracking. 
* `cell_features.py` — per-cell feature extraction & visualization. 
* `utilities.py` — label-merging helpers. 
* `requirements.txt` — pinned deps for reproducibility. 

---

### 📫 **Contact**
For questions or contributions, please contact:
**Mariia Diakova**
- GitHub: [MDyakova](https://github.com/MDyakova)
- email: m.dyakova.ml@gmail.com


python3.8 track.py --image_directory "../Avik/20250624_All_Images_for_Masha/New Turgor Analysis" --output_directory "../Avik/tracking_092225" --tile_size 400

python3.8 cell_features.py --labels_directory "../Avik/../Avik/tracking_251016/Acetate" --output_directory "../Avik/results_cells_20251030" --selected_ids "../Avik/20250624_All_Images_for_Masha/Acetate Label Selection.xlsx"


docker build -t cell_plasmolysis:v1 .
docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_plasmolysis:v1 python track.py --image_directory "/cell_tracking_files/data/Acetate" --output_directory "/cell_tracking_files/tracking_results" --tile_size 400 --name_filter roi0 --frames_exclude_file "/cell_tracking_files/ROIs to be segmented 20251029.xlsx"

docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_plasmolysis:v1 python cell_features.py --labels_directory "/cell_tracking_files/tracking_results" --output_directory "/cell_tracking_files/final_results" --selected_ids "/cell_tracking_files/Acetate Label Selection.xlsx"


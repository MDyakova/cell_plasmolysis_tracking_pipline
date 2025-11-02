import numpy as np
import glob
import argparse
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
import os
import pandas as pd
import trackpy as tp
from skimage.measure import regionprops_table
from sklearn.neighbors import NearestNeighbors

from utilities import find_similar_contours_fast

io.logger_setup()  # run this to get printing of progress


def frame_nn_p10(df):
    nn_d = []
    for f, g in df.groupby("frame"):
        if len(g) < 2:
            continue
        X = g[["x", "y"]].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        dists, _ = nbrs.kneighbors(X)
        nn_d.extend(dists[:, 1])  # distance to nearest other point
    return np.percentile(nn_d, 10) if nn_d else np.inf


def main():
    """
    Cell segmentation and tracking
    """
    # Load model
    model = models.CellposeModel(gpu=True)

    # Load input parameters
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--image_directory", type=str, help="Directory with tiff files")
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument(
        "--frames_exclude_file",
        type=str,
        default=None,
        help="Excel file with incorrect frames",
    )
    parser.add_argument("--tile_size", type=int, help="Size of one tile")
    parser.add_argument(
        "--name_filter", type=str, default="", help="Part of file name for filtration"
    )
    args = parser.parse_args()

    image_directory = os.path.abspath(args.image_directory)
    output_directory = os.path.abspath(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    tile_size = int(args.tile_size)
    name_filter = args.name_filter

    # Load list with processed files
    if os.path.exists(os.path.join(output_directory, "processed_files.txt")):
        with open(os.path.join(output_directory, "processed_files.txt"), "r") as f:
            processed_files = f.read()
        processed_files = processed_files.split("\n")
    else:
        processed_files = []

    # Load file with incorrect frames for images
    if args.frames_exclude_file is not None:
        frames_exclude_file = os.path.abspath(args.frames_exclude_file)
        incorrect_images = pd.read_excel(frames_exclude_file)
        incorrect_images = incorrect_images[
            pd.notna(incorrect_images["Frames to exclude"])
        ]

    # Go throw all subdirectories in the image_directory
    tiff_files = glob.glob(os.path.join(image_directory, "**", "*.tif"), recursive=True)
    for entire_file_name in tiff_files:
        if name_filter in entire_file_name:
            # Read image
            image = tiff.imread(entire_file_name)
            # Name to save files
            file_name_save = entire_file_name.split("/")[-1].split(".tif")[0]

            # Find all incorrect frames for image
            if args.frames_exclude_file is not None:
                incorrect_images_file = incorrect_images[
                    incorrect_images["Filename"] == entire_file_name.split("/")[-1]
                ]
                if len(incorrect_images_file) > 0:
                    frames_exclude = str(
                        incorrect_images_file["Frames to exclude"].max()
                    )
                    frames_exclude = [int(i) for i in frames_exclude.split(";")]
                    image = np.delete(image, frames_exclude, axis=0)
            # Split image to XY tiles
            for i in range(0, image.shape[1], tile_size - 50):
                for j in range(0, image.shape[2], tile_size - 50):
                    file_name = f"{file_name_save}_x_{str(j)}_y_{str(i)}.tif"
                    file_path = os.path.join(output_directory, file_name)
                    if file_name not in processed_files:
                        all_masks_save = []
                        all_flows_save = []
                        all_cells = []
                        # Segmentation for each layer in a XYZ tile
                        for step in range(image.shape[0]):
                            k = 0
                            tile_crops = []
                            diameters_results = []

                            # Make segmentation for each tile for rotation on 0, 90, 180 and 270 degrees
                            for k in [0, 1, 2, 3]:
                                # input image for model
                                image_save = Image.fromarray(
                                    np.rot90(
                                        (
                                            image[step][
                                                i : i + tile_size, j : j + tile_size
                                            ]
                                            / image[step][
                                                i : i + 400, j : j + 400
                                            ].max()
                                            * 255
                                        ),
                                        k=k,
                                    ).astype("uint8")
                                )
                                # If image size is not equal the tile size
                                if image_save.size != (tile_size, tile_size):
                                    im = (
                                        image[step][
                                            i : i + tile_size, j : j + tile_size
                                        ]
                                        / image[step][
                                            i : i + tile_size, j : j + tile_size
                                        ].max()
                                        * 255
                                    )
                                    image_new = np.zeros((tile_size, tile_size))
                                    image_new[0 : im.shape[0], 0 : im.shape[1]] = im
                                    image_save = Image.fromarray(
                                        np.rot90(image_new, k=k).astype("uint8")
                                    )

                                # Make prediction
                                masks_pred, flows, styles = model.eval(
                                    [np.array(image_save)],
                                    niter=1000,
                                    cellprob_threshold=0,
                                    diameter=40,
                                )

                                # Back rotation
                                pred = np.rot90(masks_pred[0], k=-1 * k)
                                pred = np.where(pred > 0, pred + 1000 * k, 0)

                                tile_crops.append(pred)

                            # Join segmentation for all rotated files
                            tile_crops_all = np.stack(tile_crops, axis=2)
                            all_masks = find_similar_contours_fast(tile_crops_all)
                            all_masks_save.append(all_masks)

                            # Make tabel with all segmented cells
                            props = regionprops_table(
                                all_masks, properties=("label", "centroid", "area")
                            )
                            df = pd.DataFrame(props)
                            df.rename(
                                columns={"centroid-0": "y", "centroid-1": "x"},
                                inplace=True,
                            )
                            df["frame"] = step
                            all_cells.append(df)

                        # Combine all frames
                        df_cells = pd.concat(all_cells, ignore_index=True)
                        df_cells = df_cells[df_cells["area"] > 100]
                        if len(df_cells) >= len(all_masks_save):
                            # Track using trackpy
                            try:
                                tracked = tp.link_df(
                                    df_cells, search_range=20, memory=2
                                )
                                # Remove short-lived tracks
                                tracked = tp.filter_stubs(
                                    tracked, threshold=len(all_masks_save)
                                )
                                all_masks_save = np.stack(all_masks_save)
                                tracked.reset_index(drop=True, inplace=True)

                                if len(pd.unique(tracked["label"])) == 0:
                                    tracked = tp.link_df(
                                        df_cells, search_range=30, memory=2
                                    )
                                    # Remove short-lived tracks
                                    tracked = tp.filter_stubs(
                                        tracked, threshold=len(all_masks_save)
                                    )
                                    all_masks_save = np.stack(all_masks_save)
                                    tracked.reset_index(drop=True, inplace=True)
                            except:
                                tracked = tp.link_df(
                                    df_cells, search_range=15, memory=2
                                )
                                # Remove short-lived tracks
                                tracked = tp.filter_stubs(
                                    tracked, threshold=len(all_masks_save)
                                )
                                all_masks_save = np.stack(all_masks_save)
                                tracked.reset_index(drop=True, inplace=True)

                            for frame in range(len(all_masks_save)):
                                labels_in_frame = tracked[tracked["frame"] == frame][
                                    "label"
                                ]
                                mask_frame = all_masks_save[frame]
                                mask_frame = np.where(
                                    np.isin(mask_frame, labels_in_frame), mask_frame, 0
                                )
                                all_masks_save[frame] = mask_frame

                            for ind in tracked.index:
                                label = tracked.loc[ind, "label"]
                                frame = tracked.loc[ind, "frame"]
                                cell = tracked.loc[ind, "particle"]
                                mask_frame = all_masks_save[frame]
                                mask_frame = np.where(
                                    mask_frame == label, cell, mask_frame
                                )
                                all_masks_save[frame] = mask_frame

                            if np.max(all_masks_save > 0) > 0:
                                # Save results
                                video = (
                                    image[:, i : i + tile_size, j : j + tile_size]
                                    / image[step][
                                        i : i + tile_size, j : j + tile_size
                                    ].max()
                                    * 255
                                ).astype("uint8")
                                tiff.imwrite(
                                    os.path.join(
                                        output_directory,
                                        f"{file_name_save}_x_{str(j)}_y_{str(i)}.tif",
                                    ),
                                    video.astype(np.uint16),
                                )
                                tiff.imwrite(
                                    os.path.join(
                                        output_directory,
                                        f"{file_name_save}_x_{str(j)}_y_{str(i)}_labels.tif",
                                    ),
                                    all_masks_save.astype(np.uint16),
                                )
                        with open(
                            os.path.join(output_directory, "processed_files.txt"), "a"
                        ) as f:
                            f.write(
                                f"{file_name_save}_x_{str(j)}_y_{str(i)}.tif" + "\n"
                            )


if __name__ == "__main__":
    main()

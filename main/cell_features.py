"""
Compute cell sizes
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import cv2
from skimage.transform import resize



def main():
    """
    Compute cell features for segmented cells
    """
    # Load input parameters
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument(
        "--labels_directory", type=str, help="Directory with segmented files"
    )
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument(
        "--k_pixels", type=int, default=0.1, help="Coefficient of pixels ratio"
    )
    parser.add_argument(
        "--k_intens", type=int, default=0.5, help="Coefficient of pixel intensity"
    )
    parser.add_argument(
        "--selected_ids", type=str, default=None, help="Excel file with selected labels"
    )

    args = parser.parse_args()
    labels_directory = os.path.abspath(args.labels_directory)
    output_directory = os.path.abspath(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    coeff = int(args.k_pixels)
    coeff_intens = int(args.k_pixels)

    # Load file with selected ids
    if args.selected_ids is not None:
        selected_ids_file = os.path.abspath(args.selected_ids)
        selected_ids = pd.read_excel(selected_ids_file)
        selected_ids["Filename"] = selected_ids["Filename"].fillna(method="ffill")

    # Labels for selected frames
    NUMBER_1 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]

    NUMBER_2 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1, 1.0, 1.0, 0.0],
    ]

    # Find all segmented files
    all_files = os.listdir(labels_directory)
    all_files = list(filter(lambda p: "_labels" in p, all_files))

    # Compute cell features for all segmented files
    all_sizes = []
    for file_name in all_files:
        # Create directories for output files
        folder = "_".join(file_name.split("_")[:-7])
        subfolder = file_name.split("_")[-7]
        directory = os.path.join(output_directory, folder, subfolder)
        movie_directory = os.path.join(directory, "movie")
        individual_directory = os.path.join(directory, "individual")
        before_after_directory = os.path.join(directory, "before_after")
        os.makedirs(movie_directory, exist_ok=True)
        os.makedirs(individual_directory, exist_ok=True)
        os.makedirs(before_after_directory, exist_ok=True)

        # Read image and labels files
        image = tiff.imread(
            os.path.join(labels_directory, file_name.replace("_labels", ""))
        )
        labels = tiff.imread(os.path.join(labels_directory, file_name))

        # if file is outside
        if image.shape != labels.shape:
            labels = labels[:, :image.shape[1], :image.shape[2]]

        # Check that files if correct
        if image.shape == labels.shape:
            steps_number, shape_y, shape_x = labels.shape
            # Find all labels
            all_cells = np.unique(labels)[1:]
            # Compute features only for selected files
            for cell_id in all_cells:
                # Check file is selected
                if args.selected_ids is not None:
                    is_selected = (
                        len(
                            selected_ids[
                                (
                                    selected_ids["Filename"]
                                    == file_name.split("_labels.tif")[0]
                                )
                                & (selected_ids["Label"] == cell_id)
                            ]
                        )
                        > 0
                    )
                else:
                    # If file with selected files isn't exist, process each label
                    is_selected = True
                if is_selected:
                    try:
                        k = 0
                        # Compute features for each image frames
                        for step in range(steps_number):
                            # Compute bounding box
                            y, x = np.where(labels[step] == cell_id)
                            y_min = np.maximum(0, y.min() - 5)
                            y_max = np.minimum(y.max() + 5, shape_y)
                            x_min = np.maximum(0, x.min() - 5)
                            x_max = np.minimum(x.max() + 5, shape_x)

                            # Ensure that the cell is not located near the edge.
                            if (
                                (y_min > 5)
                                & (y_max < shape_y - 5)
                                & (x_min > 5)
                                & (x_max < shape_x - 5)
                            ):
                                k += 1
                        # If cell  is not located near the edge for all frames add to list
                        # if k==steps_number:
                        # Find all saved movies
                        existing_movies = os.listdir(movie_directory)
                        # Determine next label if for saving
                        if len(existing_movies) == 0:
                            cell_id_save = 0
                        else:
                            cell_id_save = (
                                np.max(
                                    [
                                        int(i.split(".tif")[0].split("_")[-1])
                                        for i in existing_movies
                                    ]
                                )
                                + 1
                            )
                        # File name with new id
                        file_name_save = (
                            folder + "_" + subfolder + "_" + str(cell_id_save)
                        )
                        # Compute cell features
                        cell_sizes = []
                        # Create empty mask for cell.
                        # Mask bigger than cell to avoid errors with rotation
                        whole_mask = np.zeros((steps_number, 200, 200))
                        # Compute features for each image frames
                        for step in range(steps_number):
                            # Compute bounding box
                            y, x = np.where(labels[step] == cell_id)
                            y_min = np.maximum(0, y.min() - 5)
                            y_max = np.minimum(y.max() + 5, shape_y)
                            x_min = np.maximum(0, x.min() - 5)
                            x_max = np.minimum(x.max() + 5, shape_x)

                            # Locate cell and label in the center of mask
                            image_with_cell = np.where(
                                labels[step] == cell_id, image[step], 0
                            )[y_min:y_max, x_min:x_max]
                            labels_with_cell = np.where(labels[step] == cell_id, 1, 0)[
                                y_min:y_max, x_min:x_max
                            ]

                            im_shape = image_with_cell.shape
                            max_shape = np.max(im_shape)
                            image_with_cell_sq = np.zeros(
                                (max_shape + 10, max_shape + 10)
                            )
                            labels_with_cell_sq = np.zeros(
                                (max_shape + 10, max_shape + 10)
                            )
                            delta_y = max_shape - im_shape[0] + 10
                            delta_x = max_shape - im_shape[1] + 10
                            image_with_cell_sq[
                                delta_y // 2 : -delta_y // 2,
                                delta_x // 2 : -delta_x // 2,
                            ] = image_with_cell
                            labels_with_cell_sq[
                                delta_y // 2 : -delta_y // 2,
                                delta_x // 2 : -delta_x // 2,
                            ] = labels_with_cell

                            # Find contour of label
                            mask_u8 = (labels_with_cell_sq > 0).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(
                                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cnt = max(contours, key=cv2.contourArea)  # largest cell

                            # Fit ellipse to get angle
                            ellipse = cv2.fitEllipse(cnt)
                            center = ellipse[0]
                            angle = ellipse[2] - 90  # orientation angle

                            # Rotate image to make cell horizontal
                            (h, w) = labels_with_cell_sq.shape[:2]
                            center = (w // 2, h // 2)
                            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated_image = cv2.warpAffine(
                                image_with_cell_sq, matrix, (w, h)
                            )

                            # Save rotated cell to mask
                            whole_mask[step][
                                100
                                - center[1] : 100
                                - center[1]
                                + rotated_image.shape[0],
                                100
                                - center[0] : 100
                                - center[0]
                                + rotated_image.shape[1],
                            ] = rotated_image

                        # Increase size (3x)
                        # It's important to draw lines carefully.
                        arr_resized = resize(
                            whole_mask,
                            (
                                whole_mask.shape[0],
                                whole_mask.shape[1] * 3,
                                whole_mask.shape[2] * 3,
                            ),
                            order=1,
                            preserve_range=True,
                            anti_aliasing=True,
                        ).astype(whole_mask.dtype)
                        arr_resized_no_contours = arr_resized.copy()

                        # Add bounding box and average line to images
                        intencity_plots = []
                        for step, arr_i in enumerate(arr_resized):
                            y, x = np.where(arr_i > 0)
                            arr_i = arr_i[y.min() : y.max(), x.min() : x.max()]
                            x_pix = arr_i.mean(axis=0) / arr_i.mean(axis=0).max()
                            y_pix = arr_i.mean(axis=1) / arr_i.mean(axis=1).max()

                            # Filter outliers for bounding box lines
                            x_min = np.where(x_pix > coeff)[0][0]
                            x_max = np.where(x_pix > coeff)[0][-1]
                            y_min = np.where(y_pix > coeff)[0][0]
                            y_max = np.where(y_pix > coeff)[0][-1]

                            # Central line
                            y_center = (y_min + y_max) // 2
                            step_line = 2
                            gap = 2

                            # Save intensity data
                            intens_plot = arr_i[y_center, x_min:x_max].copy()
                            intens_plot_diff = intens_plot.max() - intens_plot.min()
                            intens_plot_l = intens_plot / intens_plot_diff
                            # Filter pixels with intensity less than k_intens
                            start = np.where(intens_plot_l > coeff_intens)[0][0]
                            end = np.where(intens_plot_l > coeff_intens)[0][-1]
                            length_center = (end - start + 1) / 3
                            intencity_plots.append(intens_plot)

                            # Add bounding box lines
                            arr_i[y_min, x_min:x_max] = 255
                            arr_i[y_max, x_min:x_max] = 255
                            arr_i[y_min:y_max, x_min] = 255
                            arr_i[y_min:y_max, x_max] = 255

                            # Add central line
                            for x0 in range(x_min, x_max, step_line + gap):
                                arr_i[y_center, x0 : x0 + step_line] = 255
                            arr_resized[step][
                                y.min() : y.max(), x.min() : x.max()
                            ] = arr_i

                            # Compute bounding box sizes
                            length_bb = (x_max - x_min + 1) / 3
                            width_bb = (y_max - y_min + 1) / 3

                            # Append results to list
                            all_sizes.append(
                                [
                                    file_name,
                                    cell_id_save,
                                    step,
                                    length_bb,
                                    width_bb,
                                    length_center,
                                ]
                            )
                            cell_sizes.append(
                                [
                                    file_name,
                                    cell_id_save,
                                    step,
                                    length_bb,
                                    width_bb,
                                    length_center,
                                ]
                            )

                        # Cut cell from big mask
                        _, y, x = np.where(arr_resized > 0)
                        arr_resized = arr_resized[
                            :, y.min() - 5 : y.max() + 5, x.min() - 5 : x.max() + 5
                        ]
                        arr_resized_no_contours = arr_resized_no_contours[
                            :, y.min() - 5 : y.max() + 5, x.min() - 5 : x.max() + 5
                        ]

                        # Save features for data frame
                        cell_sizes = pd.DataFrame(
                            cell_sizes,
                            columns=(
                                "file_name",
                                "cell_id",
                                "step",
                                "length_bb",
                                "width_bb",
                                "length_center",
                            ),
                        )
                        min_index = np.argmax(
                            cell_sizes["length_center"]
                            == cell_sizes["length_center"].min()
                        )
                        maximal_length = cell_sizes.loc[
                            : min_index - 1, "length_center"
                        ].max()
                        max_index = np.argmax(
                            cell_sizes["length_center"] == maximal_length
                        )

                        # Save individual images
                        _, axes = plt.subplots(
                            3,
                            1,
                            figsize=(5, 6),
                            # constrained_layout=False,
                            gridspec_kw={"height_ratios": [2, 2, 1]}
                        )
                        img = arr_resized[max_index]
                        axes[0].imshow(img[2:-2, 2:-2], cmap="gray", aspect="auto")
                        axes[0].axis("off")  # remove axis

                        img = arr_resized[min_index]
                        axes[1].imshow(img[2:-2, 2:-2], cmap="gray", aspect="auto")
                        axes[1].axis("off")  # remove axis

                        intens_plot = intencity_plots[max_index]
                        intens_plot_diff = intens_plot.max() - intens_plot.min()
                        intens_plot_l = intens_plot / intens_plot_diff
                        start = np.where(intens_plot_l > coeff_intens)[0][0]
                        end = np.where(intens_plot_l > coeff_intens)[0][-1]
                        x = np.arange(len(intens_plot))
                        axes[2].plot(x / 3, intens_plot)
                        y = np.arange(intens_plot.max() + 5)
                        x = y * 0 + start
                        axes[2].plot(x / 3, y, "k--")
                        x = y * 0 + end
                        axes[2].plot(x / 3, y, "k--")
                        axes[2].set_ylim([0, intens_plot.max() + 5])
                        axes[2].grid()

                        intens_plot = intencity_plots[min_index]
                        intens_plot_diff = intens_plot.max() - intens_plot.min()
                        intens_plot_l = intens_plot / intens_plot_diff
                        start = np.where(intens_plot_l > coeff_intens)[0][0]
                        end = np.where(intens_plot_l > coeff_intens)[0][-1]
                        x = np.arange(len(intens_plot))
                        axes[2].plot(x / 3, intens_plot)
                        y = np.arange(intens_plot.max() + 5)
                        x = y * 0 + start
                        axes[2].plot(x / 3, y, "r--")
                        x = y * 0 + end
                        axes[2].plot(x / 3, y, "r--")

                        # remove spacing between subplots
                        plt.subplots_adjust(
                            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                        )

                        plt.savefig(
                            os.path.join(individual_directory, f"{file_name_save}.jpg"),
                            dpi=300,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close()

                        # Save movie
                        arr_resized[max_index][:6, :5] = np.array(NUMBER_1) * 255
                        arr_resized[min_index][:6, :5] = np.array(NUMBER_2) * 255
                        tiff.imwrite(
                            os.path.join(movie_directory, f"{file_name_save}.tif"),
                            arr_resized,
                        )

                        # Save files before and after plasmolysis
                        before_after_ind_dir = os.path.join(
                            before_after_directory, file_name_save
                        )
                        os.makedirs(before_after_ind_dir, exist_ok=True)
                        tiff.imwrite(
                            os.path.join(before_after_ind_dir, "before.tif"),
                            arr_resized_no_contours[max_index],
                        )
                        tiff.imwrite(
                            os.path.join(before_after_ind_dir, "after.tif"),
                            arr_resized_no_contours[min_index],
                        )
                    except Exception as e:
                        print(e)

    # Save tables
    df = pd.DataFrame(
        all_sizes,
        columns=(
            "file_name",
            "cell_id",
            "step",
            "length_bb",
            "width_bb",
            "length_center",
        ),
    )
    df.to_csv(os.path.join(output_directory, "cell_sizes.csv"), index=None)


if __name__ == "__main__":
    main()

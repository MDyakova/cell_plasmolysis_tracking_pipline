# cell_plasmolysis_tracking_pipline

python3.8 track.py --image_directory "../Avik/20250624_All_Images_for_Masha/New Turgor Analysis" --output_directory "../Avik/tracking_092225" --tile_size 400

python3.8 cell_features.py --labels_directory "../Avik/../Avik/tracking_251016/Acetate" --output_directory "../Avik/results_cells_20251030" --selected_ids "../Avik/20250624_All_Images_for_Masha/Acetate Label Selection.xlsx"




docker build -t cell_plasmolysis:v1 .
docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" cell_plasmolysis:v1 python track.py --image_directory "/cell_tracking_files/data/Acetate" --output_directory "/cell_tracking_files/tracking_results" --tile_size 400 --name_filter roi0 --frames_exclude_file "/cell_tracking_files/ROIs to be segmented 20251029.xlsx"

docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" cell_plasmolysis:v1 python cell_features.py --labels_directory "/cell_tracking_files/tracking_results" --output_directory "/cell_tracking_files/final_results" --selected_ids "/cell_tracking_files/Acetate Label Selection.xlsx"


"""
main.py

Author: James Daniels
"""

# Import necessary modules
import argparse
from registration_algorithms import ICP, FGR, RANSAC, multi_scale_ICP, point_to_plane_ICP
from run_experiments import run_all_experiments
from properties import calculate_properties, read_and_display_properties, visualize_dataset_properties
from data_processing import preprocess_point_clouds
import sys
sys.path.append('./fmr')
from FMR import FMR
from results_visualize import plot_result_heatmaps, display_and_save_results, create_pivot_table, create_image_pivot_table
import pandas as pd


def parameters(argv=None):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run Experiments')

    parser.add_argument('-data', '--datasets', nargs='+', default='both', choices=['eth', 'sun3d', 'cross-source', 'both', 'all'], 
                        help='List of dataset names. "all" will use all of the datasets.')

    parser.add_argument('--algorithms', nargs='+', default=[RANSAC, ICP, multi_scale_ICP, FGR, point_to_plane_ICP, FMR],
    # parser.add_argument('--algorithms', nargs='+', default=[RANSAC],
                        help='List of algorithms. Default: RANSAC, ICP, FGR, multi_scale_ICP, point_to_plane_ICP, FMR')

    parser.add_argument('--voxelsize', default=0.02, type=float, help='Voxel size used for downsampling. Default: 0.1')

    parser.add_argument('--overlap', default = 0.5, type=float, help='Minimum percent overlap in testing, Default: 0.4 (Note use decimal percentage)')

    parser.add_argument('--range_t', default = 0.5, type=float, help='Maximum range of transformation in each axis in metres, Default = 1')

    parser.add_argument('--range_r', default = 60, type=float, help='Maximum range of rotation in degrees, Default = 90')

    parser.add_argument('--rte-thresholds', nargs='+', type=float, default=[0.6, 0.8, 1.0], 
                        help='List of RTE thresholds. Default: 0.4, 0.6, 0.8, 1.0')

    parser.add_argument('--rre-thresholds', nargs='+', type=float, default=[10, 15, 20], 
                        help='List of RRE thresholds. Default: 5, 10, 15, 20')

    parser.add_argument('-o', '-outfile', '--results-file', default='results_summary.csv', type=str, 
                        help='File to store the results summary. Default: results_summary.csv')

    args = parser.parse_args(argv)
  
    return args


def main(args):
    """
    Main function to run experiments and display/save results.
    """
    # Define list of dataset names
    # Use a predefined list of datasets
    # consider removing from all the datasets being used for training!!!
    if 'all' in args.datasets:
        args.datasets = ["cross-source-dataset", "plain", "apartment", "stairs", "hauptgebaude", 
                         "gazebo_summer", "gazebo_winter", "wood_summer", "wood_autumn", 
                         "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika", "sun3d-mit_76_studyroom-76-1studyroom2", 
                         "sun3d-hotel_umd-maryland_hotel3", "sun3d-hotel_umd-maryland_hotel1", "sun3d-hotel_uc-scan3", 
                         "sun3d-home_md-home_md_scan9_2012_sep_30", "sun3d-home_at-home_at_scan1_2013_jan_1"]
        # args.datasets = ["gazebo_winter", "hauptgebaude", "sun3d-hotel_umd-maryland_hotel3", "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika", "cross-source-dataset"]

    if 'both' in args.datasets:
        args.datasets = ["cross-source-dataset", "apartment", "gazebo_winter", "sun3d-hotel_umd-maryland_hotel3", "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]
    if 'eth' in args.datasets:
        args.datasets = ["gazebo_winter"]
        # args.datasets = ["eth_apartment"]
    if 'cross-source' in args.datasets:
        args.datasets = ["cross-source-dataset"]
    if 'sun3d' in args.datasets:
        args.datasets = ["sun3d-hotel_umd-maryland_hotel3", "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]

    # Works with data downloaded from source for easy setup
    for dataset in args.datasets:
        preprocess_point_clouds(dataset, args.overlap)

    # Run experiments
    # With arguments for voxelsixe, range_t and range_r
    run_all_experiments(args.datasets, args.algorithms, args.voxelsize, args.range_t, args.range_r, args.overlap)

    print("Analyzing experiment results...")

    # Display and save results
    # would need to change this to include the other parameters to file path
    # or change this directory look up to here?
    metrics_file_path = f"voxelsize{args.voxelsize}_rangeT{args.range_t}_rangeR{args.range_r}_overlap{args.overlap}_metrics.txt"
    display_and_save_results(args.algorithms, args.datasets, args.rte_thresholds, args.rre_thresholds, args.results_file, metrics_file_path)

    # Plot result heatmaps
    plot_result_heatmaps()

    # Read the results summary into a DataFrame
    df = pd.read_csv('results_summary.csv')

    # Create pivot tables
    create_pivot_table(df, 'results')
    create_image_pivot_table(df, 'image_results')



if __name__ == "__main__":
    # Parse command line arguments
    ARGS = parameters()

    # Run main function
    main(ARGS)

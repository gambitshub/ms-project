"""
results_visualize.py

Author: James Daniels

"""
# Import necessary modules
import numpy as np
from pathlib import Path
from tabulate import tabulate
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import six


def read_metrics_file(file_path):
    """Read a metrics file and return the processed metrics."""
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            # Read the metrics file line by line, parse each line as a JSON object
            metrics = [json.loads(line) for line in file]
        print(f"Read {len(metrics)} metrics from file {file_path}.")  # Debug print
        return metrics
    else:
        print(f"File {file_path} does not exist.")  # Debug print
        return None


def display_and_save_results(algorithms, dataset_names, rte_thresholds, rre_thresholds, results_filename, metrics_file_path):
    """
    Analyze the data and print the results in a tabular form.
    """
    # Create header for the table
    header = ["Algorithm", "Dataset", "Avg. runtime", "Avg. rmse", "Avg. rte", "Avg. rre", "Avg. mae", 
              "Avg. combined_error_metric", "Avg. total_points", "Avg. downsampled_points", "Avg. overlap"]
    
    # Extend the headers with the success rates for each pair of RTE and RRE thresholds
    for rte_thresh, rre_thresh in zip(rte_thresholds, rre_thresholds):
        header.extend([f"Success Rate ({rte_thresh}, {rre_thresh})"])

    # Initialize an empty list to store the table data
    table_data = []

    # Loop over all dataset names
    for dataset_name in dataset_names:
        # Loop over all algorithms
        for algorithm in algorithms:
            # Define path to the metrics file
            full_metrics_file_path = f"results/{dataset_name}/{algorithm.__name__}/" + metrics_file_path
            metrics = read_metrics_file(full_metrics_file_path)

            # Calculate mean of each metric
            if metrics:
                avg_values = {key: np.mean([metric[key] for metric in metrics]) for key in metrics[0].keys()}
                # avg_values = {key: np.median([metric[key] for metric in metrics]) for key in metrics[0].keys()}

                # print(f"Calculated average values for dataset {dataset_name} and algorithm {algorithm}.")  # Debug print

                # Pre-fill a row for the table with 'NA'
                row = ['NA'] * len(header)

                # Update the values in the row list
                row[:2] = [algorithm.__name__, dataset_name]
                row[2:12] = [avg_values.get(key.split(" ")[1], 'NA') for key in header[2:12]]

                # Compute success rates for each pair of RTE and RRE thresholds, add them to the row
                for rte_thresh, rre_thresh, idx in zip(rte_thresholds, rre_thresholds, range(len(rte_thresholds))):
                    success_rate = np.mean([(metric['rte'] <= rte_thresh and metric['rre'] <= rre_thresh) for metric in metrics])
                    # success_rate = np.median([(metric['rte'] <= rte_thresh and metric['rre'] <= rre_thresh) for metric in metrics])

                    row[11+idx] = success_rate  # place the success rate in the correct position in the row

                # Add the row to the table data
                table_data.append(row)

    # Print the table with tabulate
    print(tabulate(table_data, headers=header))

    # print("Length of header: ", len(header))
    # print("Length of first row in table data: ", len(table_data[0]))

    # Convert your data into DataFrame
    df = pd.DataFrame(table_data, columns=header)

    # Round the numbers
    # df = df.round(2)
    # pd.set_option('display.max_columns', None)
    # print the DataFrame
    # print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(results_filename, index=False)


def plot_bar_chart(dataset_name, table_data, metrics, algorithms, header):
    for metric in metrics:
        plt.figure(figsize=(4,6))

        # find the index of the metric in the table_data rows
        metric_index = header.index(f"Avg. {metric}") 

        # get the values for the current metric and dataset
        values = [row[metric_index] for row in table_data if row[1] == dataset_name]

        algorithm_names = [row[0] for row in table_data if row[1] == dataset_name]
        
        plt.bar(algorithm_names, values)
        plt.title(f'Avg. {metric} for different algorithms on {dataset_name}')
        plt.xlabel('Algorithm')
        plt.ylabel(f'Avg. {metric}')
        
        # Create the directory if it does not exist
        if not os.path.exists('graphresults'):
            os.makedirs('graphresults')

        # save the figure to a file
        plt.savefig(f'graphresults/{dataset_name}_{metric}_comparison.png')

        matplotlib.pyplot.close()


# def plot_line_chart(algorithm_name, metrics, dataset_names):
#     for metric in metrics:
#         plt.figure(figsize=(10,6))
#         values = [result[metric] for result in results if result['algorithm_name'] == algorithm_name]
#         plt.plot(dataset_names, values)
#         plt.title(f'{metric} for {algorithm_name} on different datasets')
#         plt.xlabel('Dataset')
#         plt.ylabel(metric)
#         plt.show()


def plot_heatmaps(table_data, header, units):
    """
    Generate heatmaps for each metric in the provided table data.

    Parameters:
    table_data: List of lists, each inner list represents a row in the table.
    header: List of strings, column names for the table data.
    units: Dictionary with metrics as keys and their units as values. 
    """

    # Convert the table data into a pandas dataframe
    df = pd.DataFrame(table_data, columns=header)

    # Convert the data types of metrics to float for visualization
    for metric in header[2:]:
        df[metric] = df[metric].astype(float)

    # Loop through each metric
    for metric in header[2:]:
        # Create a dataframe where each row is an algorithm and each column is a dataset
        heatmap_data = df.pivot(index='Algorithm', columns='Dataset', values=metric)

        # Generate a heatmap
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt=".2f")  # annot=True to add annotation to squares, fmt to control how to annotate the number

        # Set labels
        plt.title(f"Heatmap of {metric}")
        plt.xlabel('Dataset')
        plt.ylabel('Algorithm')

        # If the metric has a unit, add it to the color bar label
        cbar = plt.gcf().axes[-1]
        cbar.set_ylabel(units.get(metric, ''))

        # Rotate y-axis labels and x-axis labels for better visibility
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)  # Rotating dataset names to be at an angle for better visibility

        # This will make sure the labels fit into the figure area
        plt.tight_layout()

        # This will move the labels a bit to the left so that they align with the heatmap cells
        plt.setp(plt.gca().get_xticklabels(), ha="right", rotation_mode="anchor")

        # Save the plot
        plt.savefig(f'graphresults/heatmap_{metric}.png')

        # Clear the plot so that the next one doesn't overlap
        plt.clf()


def plot_result_heatmaps():
    """
    Generate a heatmap for each metric in the results summary.

    Parameters:
    units (Dict[str, str]): Dictionary where keys are metric names and values are units.
    """
    units = {
    'Avg. runtime': 'seconds',
    'Avg. rmse': 'units',
    'Avg. rte': 'units',
    # Add units for the rest of metrics here

    }
    # Read the results summary into a DataFrame
    df = pd.read_csv('results_summary.csv')

    # Create directory if it doesn't exist
    if not os.path.exists('graphresults'):
        os.makedirs('graphresults')

    # Loop over all metrics in the DataFrame columns
    for metric in df.columns[2:]:
        # Pivot the DataFrame to the wide format suitable for creating a heatmap
        heatmap_data = df.pivot(index='Algorithm', columns='Dataset', values=metric)

        # Generate the heatmap
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
        plt.title(f"{metric} ({units.get(metric, '')}) Comparison")
        plt.xticks(rotation=45, ha='right')

        # Save the heatmap
        plt.savefig(f'graphresults/{metric}_comparison.png', bbox_inches='tight')

        # Clear the figure after saving to prevent overlap with the next heatmap
        plt.clf()

def create_pivot_table(df, filename, index='Algorithm', columns='Dataset'):
    """
    Create a pivot table from the dataframe and save to a CSV file.

    Parameters:
    df: DataFrame to be pivoted.
    filename: Name of the file to save the pivot table to.
    index: Column to be used as index (rows) in the pivot table.
    columns: Column to be used as columns in the pivot table.
    """

    # Loop over all metrics in the DataFrame columns
    for metric in df.columns[2:]:
        # Pivot the DataFrame to the wide format suitable for creating a pivot table
        pivot_table_data = df.pivot(index=index, columns=columns, values=metric)

        # Round all data values to 4 significant figures
        pivot_table_data = pivot_table_data.round(4)

        # Save the pivot table to a CSV file
        pivot_table_data.to_csv(f'./graphresults/{filename}_{metric}_pivot_table.csv')



def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax


def create_image_pivot_table(df, filename, index='Algorithm', columns='Dataset'):
    """
    Create a pivot table from the dataframe and save to an image.

    Parameters:
    df: DataFrame to be pivoted.
    filename: Name of the file to save the pivot table to.
    index: Column to be used as index (rows) in the pivot table.
    columns: Column to be used as columns in the pivot table.
    """
    # Loop over all metrics in the DataFrame columns
    for metric in df.columns[2:]:
        # Pivot the DataFrame to the wide format suitable for creating a pivot table
        pivot_table_data = df.pivot(index=index, columns=columns, values=metric)

        # Close all existing plots
        plt.close('all')

        # Define figure size
        fig, ax = plt.subplots(figsize=(1 + len(pivot_table_data.columns)*3, 0.5 + len(pivot_table_data.index)*0.5))
        
        # Set the table
        table = ax.table(cellText=pivot_table_data.values, 
                         colLabels=pivot_table_data.columns, 
                         rowLabels=pivot_table_data.index, 
                         loc='center',
                         cellLoc = 'center', 
                         rowLoc = 'center')

        # Hide axes
        ax.axis('off')

        # Scale the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 1.5)
        
        # Auto adjust for the column headers
        plt.title(f'{metric} Pivot Table')
        plt.gcf().autofmt_xdate()

        # Save the figure
        fig.tight_layout()
        fig.savefig(f'./graphresults/{filename}_{metric}_pivot_table.png')

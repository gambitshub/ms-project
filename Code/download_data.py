"""
download_data.py

Author: James Daniels

"""
import os
import requests
from zipfile import ZipFile
import shutil
import gdown

# def download_and_extract_cross_source_dataset(url, output_dir):
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Check if the dataset already exists
#     zip_path = os.path.join(output_dir, "cross_source_dataset.zip")
#     if os.path.exists(zip_path):
#         print("Cross Source Dataset already exists, skipping download.")
#         return

#     # Download the dataset
#     print("Downloading Cross Source Dataset")
#     gdown.download(url, zip_path, quiet=False)

#     # Unzip the dataset
#     with ZipFile(zip_path, 'r') as zipObj:
#         print("Unzipping Cross Source Dataset")
#         zipObj.extractall(output_dir)

#     # Optionally, remove the zip file after extraction
#     # os.remove(zip_path)

def download_and_extract_cross_source_dataset(url, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the dataset already exists
    zip_path = os.path.join(output_dir, "cross_source_dataset.zip")
    if os.path.exists(zip_path):
        print("Cross Source Dataset already exists, skipping download.")
        return

    # Download the dataset
    print("Downloading Cross Source Dataset")
    gdown.download(url, zip_path, quiet=False)

    # Unzipping the dataset
    print("Unzipping Cross Source Dataset")

    # Extract the zip file's contents to the parent directory of output_dir
    with ZipFile(zip_path, 'r') as zipObj:
        zipObj.extractall(os.path.dirname(output_dir)) 

    # Optionally, remove the zip file after extraction
    # os.remove(zip_path)



def download_and_extract_ETH_datasets(datasets, root_dir):
    # Create root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for dataset in datasets:
        dataset_dir = os.path.join(root_dir, dataset[0])
        zip_path = os.path.join(dataset_dir, dataset[0] + ".zip")

        if os.path.exists(dataset_dir) and os.path.exists(zip_path):
            print(f"Dataset {dataset[0]} already exists, skipping download.")
            continue

        # Download dataset
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        print("Downloading dataset %s" % dataset[0])
        req = requests.get(dataset[1])
        with open(zip_path, "wb") as archive:
            archive.write(req.content)

        # Unzipping the file
        with ZipFile(zip_path, 'r') as zipObj:
            print(f"Unzipping dataset {dataset[0]}")
            zipObj.extractall(dataset_dir)

def download_and_extract_SUN3D_datasets(datasets, root_dir):
    # Create root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for dataset in datasets:
        dataset_dir = os.path.join(root_dir, dataset[0])
        zip_path = os.path.join(dataset_dir, dataset[0] + ".zip")

        if os.path.exists(dataset_dir) and os.path.exists(zip_path):
            print(f"Dataset {dataset[0]} already exists, skipping download.")
            continue

        # Download dataset
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        print("Downloading dataset %s" % dataset[0])
        req = requests.get(dataset[1])
        with open(zip_path, "wb") as archive:
            archive.write(req.content)

        # Unzipping the file
        with ZipFile(zip_path, 'r') as zipObj:
            print(f"Unzipping dataset {dataset[0]}")
            zipObj.extractall(dataset_dir)

        extracted_folder = os.path.join(dataset_dir, dataset[0])

        # Moving files from the additional directory level
        for filename in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, filename), dataset_dir)

        # Removing the now empty directory
        os.rmdir(extracted_folder)


def download_and_extract_SUN3D_eval_datasets(datasets, root_dir):
    # Create root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for dataset in datasets:
        dataset_dir = os.path.join(root_dir, dataset[0])
        target_folder = dataset[0].replace('-evaluation', '')
        target_folder_dir = os.path.join(root_dir, target_folder)
        target_gt_log_path = os.path.join(target_folder_dir, 'gt.log')

        # Check if the gt.log file already exists in the target folder
        if os.path.exists(target_gt_log_path):
            print(f"Dataset {dataset[0]} already exists, skipping download.")
            continue

        # Download dataset
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        print("Downloading dataset %s" % dataset[0])
        req = requests.get(dataset[1])
        zip_path = os.path.join(dataset_dir, dataset[0] + ".zip")
        with open(zip_path, "wb") as archive:
            archive.write(req.content)

        # Unzipping the file
        with ZipFile(zip_path, 'r') as zipObj:
            print(f"Unzipping dataset {dataset[0]}")
            zipObj.extractall(dataset_dir)

        extracted_folder = os.path.join(dataset_dir, dataset[0])

        # Moving files from the additional directory level
        for filename in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, filename), dataset_dir)
        # Removing the now empty directory
        os.rmdir(extracted_folder)

        # Moving gt.log file to respective dataset folder
        eval_gt_log_path = os.path.join(dataset_dir, 'gt.log')
        if os.path.exists(eval_gt_log_path):
            shutil.move(eval_gt_log_path, target_gt_log_path)

        # Removing the zip file and other downloaded files
        os.remove(zip_path)
        for filename in os.listdir(dataset_dir):
            if filename != 'gt.log':
                os.remove(os.path.join(dataset_dir, filename))

        # Remove the evaluation directory if it is empty
        if not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)


def main():
    ETH_datasets = [["apartment", "http://robotics.ethz.ch/~asl-datasets/apartment_03-Dec-2011-18_13_33/csv_global/global_frame.zip"],
            ["hauptgebaude", "http://robotics.ethz.ch/~asl-datasets/ETH_hauptgebaude_23-Aug-2011-18_43_49/csv_global/global_frame.zip"],
            ["stairs", "http://robotics.ethz.ch/~asl-datasets/stairs_26-Aug-2011-14_26_14/csv_global/global_frame.zip"],
            ["plain", "http://robotics.ethz.ch/~asl-datasets/plain_01-Sep-2011-16_39_18/csv_global/global_frame.zip"],
            ["gazebo_summer", "http://robotics.ethz.ch/~asl-datasets/gazebo_summer_04-Aug-2011-16_13_22/csv_global/global_frame.zip"],
            ["gazebo_winter", "http://robotics.ethz.ch/~asl-datasets/gazebo_winter_18-Jan-2012-16_10_04/csv_global/global_frame.zip"],
            ["wood_summer", "http://robotics.ethz.ch/~asl-datasets/wood_summer_25-Aug-2011-13_00_30/csv_global/global_frame.zip"],
            ["wood_autumn", "http://robotics.ethz.ch/~asl-datasets/wood_autumn_09-Dec-2011-15_44_05/csv_global/global_frame.zip"]]

    SUN3D_datasets = [["sun3d-home_at-home_at_scan1_2013_jan_1", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1.zip"],
                      ["sun3d-home_md-home_md_scan9_2012_sep_30", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30.zip"],
                      ["sun3d-hotel_uc-scan3", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3.zip"],
                      ["sun3d-hotel_umd-maryland_hotel1", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1.zip"],
                      ["sun3d-hotel_umd-maryland_hotel3", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip"],
                      ["sun3d-mit_76_studyroom-76-1studyroom2", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2.zip"],
                      ["sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip"]]

    SUN3D_eval_datasets = [["sun3d-home_at-home_at_scan1_2013_jan_1-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1-evaluation.zip"],
                           ["sun3d-home_md-home_md_scan9_2012_sep_30-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30-evaluation.zip"],
                           ["sun3d-hotel_uc-scan3-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3-evaluation.zip"],
                           ["sun3d-hotel_umd-maryland_hotel1-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1-evaluation.zip"],
                           ["sun3d-hotel_umd-maryland_hotel3-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3-evaluation.zip"],
                           ["sun3d-mit_76_studyroom-76-1studyroom2-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2-evaluation.zip"],
                           ["sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation", "http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation.zip"]]


    download_and_extract_ETH_datasets(ETH_datasets, 'data/ETH')
    download_and_extract_SUN3D_datasets(SUN3D_datasets, 'data/SUN3D')
    download_and_extract_SUN3D_eval_datasets(SUN3D_eval_datasets, 'data/SUN3D')

    cross_source_dataset_url = "https://drive.google.com/uc?id=1wuMquKHjTo7zB5vgNB8bcogYrjQn9rKS"
    download_and_extract_cross_source_dataset(cross_source_dataset_url, 'data/cross-source-dataset')


if __name__ == "__main__":
    main()


"""
                                    -- FALSE DATA ACQUISITION --
-> This script downloads the FER2013 dataset using the Kaggle CLI and processes it for our project.
-> It keeps only the neutral images from the dataset, focusing on training a personal facial recognition model.
-> The dataset will be downloaded into the 'data' folder relative to this script's location.

Prerequisites:
1. Ensure Kaggle CLI is installed: pip install kaggle
2. Log in to your Kaggle account in your browser.
3. Run this script directly to download and process the dataset.
"""


import os
import shutil

# We find data path that we are currently in and define the data_path
base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_directory, "data")
neutral_directory = os.path.join(data_path, "False")

# This text should be the command for downloading the dataset
terminal_code = f"kaggle datasets download msambare/fer2013 -p {data_path} --unzip"


def download_and_extract_dataset():
    """
    -> This function will run the "terminal_code" string in the CLI
    -> It downloads the dataset from Kaggle and saves it into "data_path" specified
    """

    print("Downloading FER2013 dataset!")
    os.system(terminal_code)
    print("Dataset downloaded!")


def create_neutral_directory():
    """
    -> This function will create a new directory into /data
    -> Because the FER2013 dataset has 7 emotions directories we want
       actually use only the neutral emotion images for our task
    """

    if not os.path.exists(neutral_directory):
        os.mkdir(neutral_directory)
        print("Created neutral(False) directory!")
    else:
        print("Neutral(False) directory already exists!")


def extract_neutral_images():
    """
    -> Look into the data directory with the FER2013 images and extract only the neutral ones
    -> Move neutral images from train directory in FER2013 to the Neutral(False) directory
    """

    train_dir = os.path.join(data_path, "train")

    if not os.path.exists(train_dir):
        print("Train directory missing from FER2013 dataset download!")
        return

    print("Extracting neutral images!")
    for label_dir in os.listdir(train_dir):
        label_path = os.path.join(train_dir, label_dir)
        if label_dir.lower() == "neutral" and os.path.isdir(label_path):
            for file in os.listdir(label_path):
                source = os.path.join(label_path, file)
                destination = os.path.join(neutral_directory, file)
                shutil.move(source, destination)
            print("Done moving neutral images into Neutral(False) Directory")
            break
    else:
        print("No neutral folder found in train directory!")


def clean_up_workspace():
    """
    -> This function is used to remove all unnecessary things left from the extract/move process
    """
    print("Cleaning Working Space!")
    shutil.rmtree(os.path.join(data_path, "train"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_path, "test"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_path, "val"), ignore_errors=True)
    print("Cleanup complete!")

def main():
    """
    -> Main script execution
    """

    # Step 1: Download the dataset
    download_and_extract_dataset()

    # Step 2: Create a directory for neutral images
    create_neutral_directory()

    # Step 3: Extract neutral images
    extract_neutral_images()

    # Step 4: Clean up
    clean_up_workspace()

if __name__ == "__main__":
    main()




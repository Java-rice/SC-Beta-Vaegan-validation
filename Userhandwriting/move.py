import os
import shutil

def copy_svc_files(source_dir, destination_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # Check for .svc files
            if filename.endswith(".svc"):
                source_file = os.path.join(root, filename)
                destination_file = os.path.join(destination_dir, filename)
                shutil.copy(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")

# Use the exact absolute path for the source and destination directories
source_directory = r'F:\Users\katod\OneDrive - Itech\NEW Backup\Downloads\AllData (1)\DataEmothaw\Collection1'
destination_directory = r'C:\Github\SC-Beta-Vaegan-validation\Userhandwriting\Collection1'  # Update this with your exact destination path

copy_svc_files(source_directory, destination_directory)

import os
import shutil

def copy_svc_files(source_dir, destination_dir):

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, files in os.walk(source_dir):
        for filename in files:

            if filename.endswith(".svc"):

                source_file = os.path.join(root, filename)
                destination_file = os.path.join(destination_dir, filename)
                shutil.copy(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")


source_directory = './Collection1'
destination_directory = './destination'

copy_svc_files(source_directory, destination_directory)

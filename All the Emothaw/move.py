import os
import shutil

def copy_svc_files(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Function to create a new file name if a conflict occurs
    def get_unique_filename(dest_path, filename):
        base, ext = os.path.splitext(filename)
        counter = 2
        new_filename = filename
        # Check for existing file and create a new name with counter if needed
        while os.path.exists(os.path.join(dest_path, new_filename)):
            new_filename = f"{base}({counter}){ext}"
            counter += 1
        return new_filename

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.svc'):
                # Construct full file path
                file_path = os.path.join(root, file)
                
                # Get a unique file name if there's a conflict
                dest_file = get_unique_filename(dest_dir, file)
                
                # Copy file to destination (ignoring folder structure)
                shutil.copy(file_path, os.path.join(dest_dir, dest_file))
                print(f"Copied: {file_path} to {os.path.join(dest_dir, dest_file)}")

# Example usage
source_directory = './data/DataEmothaw'
destination_directory = './compileddata'

copy_svc_files(source_directory, destination_directory)

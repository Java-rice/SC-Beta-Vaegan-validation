import os

def rename_files(directory, label):
    # Get all files in the specified directory
    for filename in os.listdir(directory):
        # Check if it's a file
        if os.path.isfile(os.path.join(directory, filename)):
            # Remove underscores from the filename
            new_filename = filename.replace("_", "")
            # Add the label with an underscore prefix
            new_filename = f"{label}_{new_filename}"
            # Rename the file
            os.rename(
                os.path.join(directory, filename),
                os.path.join(directory, new_filename)
            )
            print(f"Renamed '{filename}' to '{new_filename}'")

# Usage
directory_path = "./user00045/session00001"  # Replace with the directory path
label = "0"  
rename_files(directory_path, label)
import os

def export_source_code(root_folder, output_file, excluded_folders=None):
    """
    Exports all Python source code from a given root folder to a single text file.

    Args:
        root_folder (str): The root directory of the project.
        output_file (str): The path to the output text file.
        excluded_folders (list, optional): A list of folder names to exclude.
                                         Defaults to None.
    """
    if excluded_folders is None:
        excluded_folders = []

    # Use 'a' mode for append, so we don't overwrite previous content
    with open(output_file, 'a', encoding='utf-8') as outfile:
        # Walk through the directory tree
        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Modify dirnames in place to prevent os.walk from entering excluded folders
            dirnames[:] = [d for d in dirnames if not any(os.path.join(dirpath, d).startswith(os.path.normpath(f)) for f in excluded_folders)]

            # Iterate over the files in the current directory
            for filename in filenames:
                if filename.endswith(".py"):
                    # Construct the full file path
                    filepath = os.path.join(dirpath, filename)
                    # Normalize the path for a consistent output format
                    relative_filepath = os.path.relpath(filepath, '.')

                    # Write the file header
                    outfile.write(f"- {relative_filepath}\n")
                    outfile.write('"""\n')

                    # Write the file content
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        print(f"Could not read file {filepath}: {e}")

                    # Write the file footer
                    outfile.write('\n"""\n')
                    outfile.write('\n')

if __name__ == '__main__':
    # Define your project source folders and output file
    # You can add more folders to this list if needed
    source_folders = ["src", "node_wrappers"]
    output_filename = "source.txt"

    # Define any folders you want to exclude
    # Use paths relative to the project root
    folders_to_exclude = [
        "src/mediapipe_vision",
    ]

    # First, clear the existing output file by opening it in write mode ('w')
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('')

    # Then, iterate through the list of source folders and export the code
    for folder in source_folders:
        export_source_code(folder, output_filename, folders_to_exclude)

    print(f"All source code from {source_folders} exported to {output_filename}")
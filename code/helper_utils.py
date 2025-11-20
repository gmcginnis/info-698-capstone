import os
# The data is located in a different folder from the cwd
def get_path(input_filename, input_parent = "data"):
    """Get the file path for specified data file.

    Args:
        input_filename (str): Data file name in the ~/input_parent folder.
        input_parent (str): Parent folder name, defaults to 'data'.

    Returns:
        str: Full absolute path to file.
    """
    path_data = os.path.abspath(os.path.join('..', input_parent, input_filename))
    return path_data
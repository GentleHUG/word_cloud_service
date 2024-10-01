import os


def read_file_lines(filepath):
	"""
    Reads a text or CSV file and returns a list of lines.

    :param filepath: Path to the .txt or .csv file.
    :return: List of lines in the file.
    :raises ValueError: If the file format is not supported.
    :raises IOError: If there is an issue opening or reading the file.
    """
	_, file_extension = os.path.splitext(filepath)
	if file_extension not in ['.txt', '.csv']:
		raise ValueError("Unsupported file format. Please provide a .txt or .csv file.")

	try:
		with open(filepath, 'r', encoding='utf-8') as file:
			lines = file.readlines()
		return [line.strip() for line in lines]
	except IOError as e:
		raise IOError(f"Error reading file: {e}")

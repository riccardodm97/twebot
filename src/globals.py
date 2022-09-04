from pathlib import Path

# PATHS 
BASE_PATH = Path().absolute()              #Path(*Path().absolute().parts[:-1]) for jupyter 
DATA_FOLDER = BASE_PATH / 'data' # directory containing the notebook


force_processing = False 
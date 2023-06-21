import os
from pathlib import Path
path_folder = os.path.dirname(__file__)

print(Path(path_folder).joinpath('scr/data/save_models'))



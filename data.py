import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64] 


@dataclasses.dataclass
class ModelInputs:
    # This should contain what were previously named rho, mu_F,
    # and the other opaque names as objects of type FloatArray
    pass


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Vernon, maybe do this one? Should be reasonably simple data handling stuff
    games = upcoming = seasons = pd.DataFrame(None)
    return games, upcoming, seasons


def build_inputs(
    games: pd.DataFrame, upcoming: pd.DataFrame, seasons: pd.DataFrame
) -> ModelInputs:
    # Generate a ModelInputs from the three dataframes we get from load_data
    return ModelInputs

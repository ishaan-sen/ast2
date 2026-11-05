import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64] 

@dataclasses.dataclass
class ModelInputs:
    #Season
    carryover: FloatArray #Parameter 1: Share of strength carried into next season (SP+)
    offseasonChange: FloatArray #Parameter 2: Beginning of season uncertianty (SP+)
    #Game
    site: FloatArray #Parameter 3: site; home (1), away (-1), neutral (0)
    opponentELO: FloatArray #Parameter 4: ELO rating of the opposing team (from data)
    longRunStrength: FloatArray #Parameter 5: Long-run average team strength (SP+)
    gameNoise: FloatArray #Parameter 6: Overall game noise, variance


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Vernon, maybe do this one? Should be reasonably simple data handling stuff
    games = upcoming = seasons = pd.DataFrame(None)
    return games, upcoming, seasons


def build_inputs(
    games: pd.DataFrame, upcoming: pd.DataFrame, seasons: pd.DataFrame
) -> ModelInputs:
    # Generate a ModelInputs from the three dataframes we get from load_data
    return ModelInputs

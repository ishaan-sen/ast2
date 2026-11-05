# import numpy as np
import pandas as pd
from jax import (
    Array,  # this instead of numpy because of the numpyro integration. perhaps this is overkill.
)


def summarize_predictions(
    samples: dict[str, Array], n_played: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # I think this signature is right?
    # Split by played and predicted matches, unless we want to represent
    # that as a single dataframe and a split index throughout. 
    pass

def evaluate():
    # what do we want to do here?
    pass 

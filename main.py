import sys
from pathlib import Path

# apparently arviz is good for bayesian visualizations,
# not sure what it does quite yet. feel free to replace with something else
# if you have specific other packages in mind
import arviz as az

# all these are just gonna get tied together in main()
from data import build_inputs, load_data
from evaluate import evaluate, summarize_predictions
from model import PARAMS, run_mcmc



def main(data_dir: Path) -> None:
    # load data, build model, evaluate model, write results
    out = Path("output") # write results to this directory
    pass


if __name__ == "__main__":
    main(Path(sys.argv[1]))

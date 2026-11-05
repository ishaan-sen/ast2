import numpyro

N_CHAINS: int = 4
numpyro.set_host_device_count(
    N_CHAINS
)  # this must run before JAX initializes its backend, apparently
numpyro.enable_x64()  # misleading name, apparently this means 64 bit precision and not the x86_64 architecture

import jax

# import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS

from data import ModelInputs

PARAMS: list[str] = []  # list of named parameters for the model

Carry = tuple[Array, Array]  # (strength mean, strength variance)
Row = tuple[
    Array, ...
]  # we have seven parameters in the old model, but this should technically be generic.

def model_flat( #Model containing flat priors for all parameters
    carryover: Arraylike,
    offseasonChange: Arraylike,
    site: Arraylike,
    opponentELO: Arraylike,
    longRunStrength: Arraylike,
    gameNoise: Arraylike,
) -> None:
    # Flat distributions for all priors for first model
    p_carryover = numpyro.sample("carryover", dist.Uniform(0.0, 1.0))
    p_offseasonChange = numpyro.sample("offseasonChange", dist.Uniform(0.0, 1.0))
    p_site = numpyro.sample("site", dist.Uniform(0.0, 1.0))
    p_opponentELO = numpyro.sample("opponentELO", dist.Uniform(0.0, 1.0))
    p_longRunStrength = numpyro.sample("longRunStrength", dist.Uniform(0.0, 1.0))
    p_gameNoise = numpyro.sample("gameNoise", dist.Uniform(0.0, 1.0))

    def step(carry: Carry, row: Row) -> tuple[Carry, tuple[Array, Array, Array]]:
        
        pass

def model(
    margin: ArrayLike,
    opp_sp: ArrayLike,
    site: ArrayLike,
    new_season: ArrayLike,
    observed: ArrayLike,
    fit: ArrayLike,
    start_strength: float,
) -> None:
    # this is all stuff from chris' old model. we still need explanations for them.
    rho = numpyro.sample("rho", dist.Beta(10.0, 2.0))
    tau = numpyro.sample("tau", dist.TruncatedNormal(8.7, 3.0, low=0.0))
    q = numpyro.sample("q", dist.HalfNormal(3.0))
    mu_F = numpyro.sample("mu_F", dist.Normal(10.0, 10.0))
    k = numpyro.sample("k", dist.Normal(0.78, 0.10))
    h = numpyro.sample("h", dist.Normal(2.8, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(20.0))

    def step(carry: Carry, row: Row) -> tuple[Carry, tuple[Array, Array, Array]]:
        pass

    # do some kind of jax.lax.scan() using step, as defined above, to carry forward information. Chris, want to take this one?


def run_mcmc(inputs: ModelInputs) -> MCMC:
    mcmc = MCMC(
        NUTS(model), # fill in the other parameters here
    )
    mcmc.run(jax.random.PRNGKey(314159), **inputs) # they expect a specific PRNGKey class and not just an int or something
    return mcmc

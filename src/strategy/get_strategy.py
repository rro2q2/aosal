# Import query strategies
from strategy.query_stategy.random import RandomSampling
from strategy.query_stategy.entropy import EntropySampling
from strategy.query_stategy.bertkm import BertKMSampling
from strategy.query_stategy.badge import BADGESampling
from strategy.query_stategy.cal import CALSampling
from strategy.query_stategy.aosal import AOSALSampling

def get_sampling_strategy(cfg, model):
    if cfg.strategy == "random":
        return RandomSampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy
        )
    elif cfg.strategy == "entropy":
        return EntropySampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy
        )
    elif cfg.strategy == "bertkm":
        return BertKMSampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy
        )
    elif cfg.strategy == "badge":
        return BADGESampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy
        )
    elif cfg.strategy == "cal":
        return CALSampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy,
            k=cfg.k
        )
    elif cfg.strategy == "aosal":
        return AOSALSampling(
            model=model, 
            budget_percent=cfg.budget_percent, 
            acquisition_percent=cfg.acquisition_percent, 
            name=cfg.strategy,
            distance=cfg.distance,
            inf_measure=cfg.inf_measure
        )
    else:
        raise ValueError(f"Invalid argument value `{cfg.strategy}` for cfg.strategy")

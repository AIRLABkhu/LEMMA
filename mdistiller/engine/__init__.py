from .trainer import BaseTrainer, CRDTrainer, AugTrainer, CondenseTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer,
    "condense": CondenseTrainer,
}

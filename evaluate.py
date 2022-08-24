from pathlib import Path

import ray
import hydra
from omegaconf import DictConfig

from sklearn.pipeline import Pipeline

from src import utils
from src.training import train_and_test, split_data

log = utils.get_logger(__name__)


def instantiate_hparams(config):
    reduction = hydra.utils.instantiate(config.reduction, _convert_="partial")
    classifier = hydra.utils.instantiate(config.classifier, _convert_="partial")
    all_hparams = []

    for red in reduction:
        for cl in classifier:
            all_hparams.append(red | cl)

    return all_hparams



@hydra.main(config_path="config", config_name="config.yaml")
def run(config: DictConfig) -> None:
    ray.init(ignore_reinit_error=True, num_cpus=24)

    species = hydra.utils.instantiate(config.dataset.species)
    log.info(f"Evaluating {species}.")

    seed = config.get('seed')
    utils.seed_everything(seed)
    log.info(f"Random seed set to {seed}")

    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset = hydra.utils.instantiate(config.dataset, _convert_="partial")

    log.info(f"Instantiating pipeline")
    pipeline = Pipeline([
        ('scaling', 'passthrough'),
        ('reduction', 'passthrough'),
        ('clf', 'passthrough')
    ])

    log.info(f"Instantiating hyper-parameters")
    hparams = instantiate_hparams(config)

    # Applies optional utilities
    utils.extras(config)

    log.info(f"Instantiating tuner")
    tuner = hydra.utils.instantiate(
        config.tuner,
        estimator=pipeline,
        param_distributions=hparams,
        cv=split_data(dataset),
        _convert_="partial"
    )

    if not Path(f"{seed:02d}_ranking.csv").exists():
        test_probas, predict_probas = train_and_test(
            dataset=dataset.shuffle(),
            tuner=tuner,
            path_prefix=f"{seed:02d}_",
        )
        test_probas = test_probas.sort_values("prob", ascending=False)
        test_probas.to_csv(f"{seed:02d}_test.csv", index=False)

        predict_probas = predict_probas.sort_values("prob", ascending=False)
        predict_probas.to_csv(f"{seed:02d}_ranking.csv", index=False)


if __name__ == "__main__":
    run()

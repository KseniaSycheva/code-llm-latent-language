from datasets import load_dataset, Dataset
from omegaconf import DictConfig


def prepare_human_eval(config: DictConfig):
    tasks = []

    for lang in config.langs:
        dataset = load_dataset("zai-org/humaneval-x", lang, split="test")
        subset = dataset.shuffle(seed=config.seed).select(
            [i for i in range(config.n_lang_examples)]
        )
        tasks.extend(subset.to_list())

    return Dataset.from_list(tasks)

import json
import os

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from code_llm_latent_language.data import prepare_dataset
from code_llm_latent_language.patches import LatentsType
from code_llm_latent_language.patches.qwen import CustomQwen2ForCausalLM
from code_llm_latent_language.patches.deepseek import CustomDeepseekV3ForCausalLM


def generate_completions(
    model, tokenizer, data: Dataset, layer_index: int, latent_type
):
    generations = []

    for row in data:
        inputs = tokenizer(row["prompt"], return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")

        output = model.generate(
            **inputs,
            layer_index=layer_index,
            latent_type=latent_type,
            max_new_tokens=256,
            do_sample=True,
        )
        generations.append(
            {
                "completion": tokenizer.decode(output[0], skip_special_tokens=True),
                "prompt": row["prompt"],
            }
        )

    return generations


def prepare_model(config: DictConfig):
    if config.model_type == "qwen":
        return CustomQwen2ForCausalLM.from_pretrained(config.model_path)
    if config.model_type == "deepseek":
        return CustomDeepseekV3ForCausalLM.from_pretrained(config.model_path)

    raise ValueError(f"Model type {config.model_type} is unknown.")


@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    data = prepare_dataset(config.data)
    model = prepare_model(config)
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    os.makedirs(config.output_path, exist_ok=True)
    latent_type = LatentsType[config.latent_type]

    for layer in config.layers:
        generations = generate_completions(model, tokenizer, data, layer, latent_type)
        path = os.path.join(config.output_path, f"{layer}.json")
        with open(path, "w") as f:
            json.dump(generations, f)
        print("Saved:", os.path.abspath(path))


if __name__ == "__main__":
    main()

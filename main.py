import json
import os

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from code_llm_latent_language.data import prepare_human_eval
from code_llm_latent_language.patches import LatentsType
from code_llm_latent_language.patches.qwen import CustomQwen2ForCausalLM


def generate_completions(
    model, tokenizer, data: Dataset, layer_index: int, latent_type: LatentsType
):
    generations = []

    for row in data:
        inputs = tokenizer(row["prompt"], return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")

        output = model.generate(
            **inputs, layer_index=layer_index, latent_type=latent_type
        )
        generations.append(
            {
                "completion": tokenizer.decode(output[0], skip_special_tokens=True),
                "prompt": row["prompt"],
            }
        )

    return generations


@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    data = prepare_human_eval(config.data)
    model = CustomQwen2ForCausalLM.from_pretrained(config.model_path)
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

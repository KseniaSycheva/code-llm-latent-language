# code-llm-latent-language
A logit lens analysis of how multi-language code models process and generate programming solutions across different languages.

## Setup 
```
pip install poetry
poetry install

```

## Run 
Right now two families of models are supported: Qwen-Coder and Deepseek-Coder. Model and dataset are specified in [configs](configs).
To run experiment:
```
poetry run python main.py 

```
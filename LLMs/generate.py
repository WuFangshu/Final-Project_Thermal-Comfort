import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import FeatureProjector
from dataset import encode_sample  # we assume this function is shared in dataset.py

def generate_description(x_vector, projector, language_model, tokenizer, device):
    """
    Given one input vector, generate a natural language description using the frozen LLM.
    """
    projector.eval()
    language_model.eval()

    with torch.no_grad():
        # Generate prompt
        prompt = "Describe the thermal and emotional state: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Project the input (not directly injected for now, placeholder)
        projected = projector(x_vector.unsqueeze(0).to(device))

        # Call language model to generate text
        output = language_model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    # ----- Config -----
    model_name = "Qwen/Qwen1.5-0.5B"
    input_dim = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load model and tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    language_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # ----- Load projector -----
    projector = FeatureProjector(input_dim=input_dim, embed_dim=language_model.config.hidden_size)
    projector.load_state_dict(torch.load("projector.pth", map_location=device))
    projector.to(device)

    # ----- Example: Load one sample -----
    import pandas as pd
    df = pd.read_csv("autotherm_sample.csv")
    row = df.iloc[0]  # take first sample
    x_vector = torch.tensor(encode_sample(row), dtype=torch.float32).to(device)

    # ----- Generate -----
    result = generate_description(x_vector, projector, language_model, tokenizer, device)
    print("Generated Description:\n", result)

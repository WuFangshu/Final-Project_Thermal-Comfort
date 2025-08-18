import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import AutoThermTextDataset
from model import FeatureProjector, ProjectorTextModel

def train():
    # ----- Config -----
    model_name = "Qwen/Qwen1.5-0.5B"  # or TinyLlama
    csv_path = "autotherm_sample.csv"
    batch_size = 8
    num_epochs = 5
    lr = 1e-4
    input_dim = 80  # depends on your feature encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load model and tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    language_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # ----- Build dataset -----
    dataset = AutoThermTextDataset(csv_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----- Build projector and model -----
    projector = FeatureProjector(input_dim=input_dim, embed_dim=language_model.config.hidden_size)
    model = ProjectorTextModel(projector, language_model).to(device)

    # ----- Optimizer -----
    optimizer = AdamW(model.projector.parameters(), lr=lr)

    # ----- Training loop -----
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, input_ids, labels in dataloader:
            x = x.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = model(x, input_ids, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} / {num_epochs} - Loss: {avg_loss:.4f}")

    # ----- Save projector -----
    torch.save(model.projector.state_dict(), "projector.pth")
    print("Projector saved to projector.pth")

if __name__ == "__main__":
    train()

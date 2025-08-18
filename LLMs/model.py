import torch
import torch.nn as nn
from transformers import PreTrainedModel

class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int = 80, embed_dim: int = 512):
        """
        Maps input feature vector to embedding space compatible with a language model.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        Input: x - shape [batch_size, input_dim]
        Output: projected feature - shape [batch_size, embed_dim]
        """
        return self.net(x)


class ProjectorTextModel(nn.Module):
    def __init__(self, projector: FeatureProjector, language_model: PreTrainedModel):
        """
        Combines the feature projector with a frozen language model.
        Only the projector is trainable.
        """
        super().__init__()
        self.projector = projector
        self.language_model = language_model

        # Freeze all LLM parameters
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, x, input_ids, labels):
        """
        x: [batch_size, input_dim]       - AutoTherm feature vectors
        input_ids: [batch_size, seq_len] - tokenized prompt
        labels: [batch_size, seq_len]    - target text token IDs
        """
        _ = self.projector(x)  # Projected embedding is ignored in this baseline
        # For now, only pass prompt → labels to LLM. Conditioning can be added later.
        output = self.language_model(
            input_ids=input_ids,
            labels=labels
        )
        return output.loss

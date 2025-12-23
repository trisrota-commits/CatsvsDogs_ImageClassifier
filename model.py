import torch
import torch.nn as nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision.models as models
from huggingface_hub import hf_hub_download

def load_model():
    model_path = hf_hub_download(
        repo_id="Trisrota/cats-vs-dogs-resnet18",
        filename="cats_vs_dogs_resnet18.pth"
    )

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model

#Architecture matches training exactly

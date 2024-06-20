import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5) # Attribute num
        )

    def forward(self, x):
        return self.layers(x)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as T
import numpy as np
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self, model_path=None):
        super(FeatureExtractor, self).__init__()
        base_model = resnet18(weights=None)  # Don't load default weights
        self.model = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer

        if model_path:
            print("[INFO] Using model:", model_path)
            state_dict = torch.load(model_path, map_location='cpu')

            # ⚠️ Remove fc.* keys to avoid loading classification head
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
            self.model.load_state_dict(filtered_state_dict, strict=False)

        self.model.eval()  # Set to eval mode

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # H x W
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)

    def extract_features(self, frame, boxes):
        crops = []
        for x, y, w, h in boxes:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_tensor = self.transforms(crop).unsqueeze(0)
            crops.append(crop_tensor)

        if not crops:
            return np.array([])

        batch = torch.cat(crops, dim=0)
        with torch.no_grad():
            features = self.forward(batch)
        return features.cpu().numpy()

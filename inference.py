from Models.ResNet_RS import ResNet_RS
from Models.Gru import EfficientGRUModel
from Classifiers.FrameClassifier import FrameClassifier
from Loaders.Dataloader import DataLoaderLite
import torch
import numpy as np
import pandas as pd

conv  = ResNet_RS()
rnn = EfficientGRUModel(input_size=2048)
model = FrameClassifier(conv, rnn)

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device_type}")
model.to(device_type)

model = torch.compile(model)

checkpoint = torch.load("model_ema.pth", map_location = device_type)
model.load_state_dict(checkpoint)

model.eval()

batch_size = 16
max_step = 100 # around 1 epoch

loader = DataLoaderLite(B = batch_size)
prediction = []
correct = torch.zeros((), dtype=torch.long, device=device_type)

gru_hidden = None

for step in range(max_step):
    x, y = loader.next_batch()
    x, y = x.to(device_type), y.to(device_type)

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, h, _ = model(x, h = gru_hidden)
            gru_hidden = h

    pred = logits.argmax(dim = 1) # (B,)
    prediction.append(pred)

    correct += (pred == y).sum()

out = torch.cat(prediction, dim=0)

accuracy = (correct.float() / 1600).item()
print(f"accuracy is {accuracy:.4f}")

out = out.cpu().numpy()
df = pd.DataFrame(out)
df.to_csv("out_tensor.csv", index=False)





    





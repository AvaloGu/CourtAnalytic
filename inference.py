from Models.ResNet_RS import ResNet_RS
from Models.ConvNeXt import ConvNeXt
from Models.Gru import EfficientGRUModel, GRUStage2
from Classifiers.FrameClassifier import FrameClassifier, ShotPredictor
from Loaders.Dataloader import DataLoaderLite, DataLoaderStage2
import torch
import numpy as np
import pandas as pd

STAGE2 = True

# conv  = ResNet_RS()
# rnn = EfficientGRUModel(input_size=2048)

conv = ConvNeXt()

if STAGE2:
    rnn = GRUStage2()
    model = ShotPredictor(conv, rnn)
else:
    model = FrameClassifier(conv)


device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device_type}")
model.to(device_type)

model = torch.compile(model)

checkpoint = torch.load("model_ema.pth", map_location = device_type)
model.load_state_dict(checkpoint)

model.eval()

batch_size = 16
max_step = 100 # around 1 epoch

if STAGE2:
    loader = DataLoaderStage2() 
else:
    loader = DataLoaderLite(B = batch_size, shuffle=False)

if STAGE2:
    max_step = len(loader.target)

prediction = []
correct = torch.zeros((), dtype=torch.long, device=device_type)

# gru_hidden = None
num_shots = 0

for step in range(max_step):
    x, y = loader.next_batch()
    x, y = x.to(device_type), y.to(device_type)

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(x, y)

    pred = logits.argmax(dim = 1) # (B,)
    prediction.append(pred)

    if STAGE2:
        point_token = torch.tensor([13], dtype=torch.long, device=pred.device) # point token z
        prediction.append(point_token)
        num_shots += len(y)

    correct += (pred == y).sum()

if STAGE2:
    accuracy = (correct.float() / num_shots).item()
else:
    accuracy = (correct.float() / 1600).item()
print(f"accuracy is {accuracy:.4f}")

out = torch.cat(prediction, dim=0).cpu() # (num_of_examples,)
out = out.numpy().astype(int)

if STAGE2:
    target = np.loadtxt("target_stage2.csv", dtype=str)
    itos = {i:ch for i, ch in enumerate(np.sort(np.unique(target)))}
    decode = [itos[i] for i in out]
    df = pd.DataFrame(decode, columns=["predictions"])
else:
    df = pd.DataFrame(out)

df.to_csv("out_tensor.csv", index=False)





    





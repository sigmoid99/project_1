import torch

def predict(model, x):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    return prediction

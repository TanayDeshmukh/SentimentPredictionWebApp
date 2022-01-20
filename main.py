import os
import torch
import config
import schemas

from model import DISTILBERTUncased
from fastapi import FastAPI, Request

app = FastAPI()

model = DISTILBERTUncased().to(config.DEVICE)
model.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, "model.pth"), map_location=torch.device(config.DEVICE)))
model.eval()
print("Model loaded...")

def sentiment_prediction(sentence):
    tokenizer = config.tokenizer
    inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens = True,
            truncation=True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    padding_length = config.MAX_LEN - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    outputs = model(
        input_ids=ids,
        attention_mask=mask
    )

    outputs = torch.sigmoid(outputs)
    return outputs[0][0]

@app.get("/")
def predict(request: Request):

    return "Sentiment Analysis using DistilBERT"

@app.get("/predict/{content}", response_model=schemas.Prediction)
def predict(request: Request, content: str):

    pred = sentiment_prediction(content).item()
    return {"txt": content,
                    "positive_sentiment": str(pred),
                    "negative_sentiment": str(1 - pred)}
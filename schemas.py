from pydantic import BaseModel

class UserText(BaseModel):
    content: str

class Prediction(BaseModel):
    txt: str
    positive_sentiment: str
    negative_sentiment: str

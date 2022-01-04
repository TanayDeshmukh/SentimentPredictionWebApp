from transformers import DistilBertTokenizer

EPOCHS = 5
MAX_LEN = 512
DEVICE = "cpu"
RANDOM_SEED = 123
MODEL_PATH = "./saved_model"
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
TRAINING_FILE = './data/dataset.csv'
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased', 
    do_lower_case=True
)
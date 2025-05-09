import json
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import preprocess, clean_text
from custom_trainer import CustomTrainer
from evaluate import load  
from PIL import Image  
from transformers import TrainerCallback
from transformers import DataCollatorForSeq2Seq

# Callback untuk mencatat loss
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

# Inisialisasi callback
loss_logger = LossLoggerCallback()

# Load data
with open("dataset/data-kemasan.json") as f:
    data = json.load(f)

def clean_text_safe(text):
    text = text.encode("utf-8", "replace").decode("utf-8")
    return text

image_paths, texts = [], []
for item in data:
    image_paths.append(f"dataset/{item['image']}")
    texts.append(clean_text_safe(item["text"]))

# data collator untuk input image
def custom_data_collator(features):
    pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
    labels = [f["labels"] for f in features]
    max_length = max(len(l) for l in labels)
    padded_labels = torch.full((len(labels), max_length), fill_value=-100)
    
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = torch.tensor(l)
    
    return {"pixel_values": pixel_values, "labels": padded_labels}



# Convert data to Dataset
dataset = Dataset.from_dict({"image_path": image_paths, "text": texts})

# Split dataset into train and test (80% train, 20% test)
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

train_dataset.to_json("dataset/train_data.json")
test_dataset.to_json("dataset/test_data.json")

train_dataset = train_dataset.map(preprocess, remove_columns=["image_path", "text"])
test_dataset = test_dataset.map(preprocess, remove_columns=["image_path", "text"])


# Load model & processor
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load evaluation metric
cer = load("cer")

data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

# Training settings
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetune",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_total_limit=1,
)

# Trainer setup

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.image_processor,
    data_collator=custom_data_collator,           
    callbacks=[loss_logger],
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./trocr_finetune")

print("Training selesai dan model disimpan di ./trocr_finetune")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_logger.losses, label="Training Loss")
plt.xlabel("Logging Step")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

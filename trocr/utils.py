from transformers import TrOCRProcessor
from PIL import Image
import re

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text

def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    text = clean_text(example["text"])
    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
    labels = processor.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

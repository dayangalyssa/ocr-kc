from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
from utils import preprocess 
import evaluate
import torch

# Load CER and WER metrics
cer = evaluate.load("cer")
wer = evaluate.load("wer")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetune").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

test_data = Dataset.from_json("dataset/test_data.json")

preds, refs = [], []

for item in test_data:
    img = Image.open(item["image_path"]).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"[PREDIKSI] {pred}")
    print(f"[REFERENSI] {item['text']}")
    print("-" * 80)

    preds.append(pred)
    refs.append(item["text"])
    
# Calculate and print CER and WER
print("CER:", cer.compute(predictions=preds, references=refs))
print("WER:", wer.compute(predictions=preds, references=refs))

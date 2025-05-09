from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load model and processor for prediction
model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetune")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Use the trained model for prediction on a new image
image = Image.open("dataset/kemasan_1.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Predicted:", text)

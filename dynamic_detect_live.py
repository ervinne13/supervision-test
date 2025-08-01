
import torch
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
import argparse


# https://huggingface.co/google/owlvit-base-patch32
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Current running the thing on a mac so we can't CUDA.
# Uncomment this later when we move this to the ubuntu machine
# model.to("cuda")

parser = argparse.ArgumentParser(description="OwlViT live detection")
# 0, 1, 2, depending on the camera setup. 1 is what worked for apple continuity camera
parser.add_argument('--camera', type=int, default=1, help='Camera index for cv2.VideoCapture')
parser.add_argument('--prompt', type=str, required=True, help='Comma-separated detection prompts, e.g. "a person, digital watch"')
args = parser.parse_args()

prompt = [p.strip() for p in args.prompt.split(",") if p.strip()]

cap = cv2.VideoCapture(args.camera)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Prepare inputs
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs
    target_sizes = torch.tensor([pil_image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    # Draw bounding boxes and labels
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        box = box.int().numpy()
        label_text = prompt[label]
        score_text = f"{score:.2f}"
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_text} {score_text}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("OwlViT Detection", frame)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import gradio as gr
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image
import numpy as np

# Load model and processor
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

def detect_objects(frame, prompt):
    if frame is None or prompt.strip() == "":
        return frame
    # Support multiple prompts separated by comma
    prompts = [p.strip() for p in prompt.split(",") if p.strip()]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) if frame.shape[2] == 3 else Image.fromarray(frame)
    inputs = processor(text=prompts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
    # Draw bounding boxes and labels
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        box = box.int().numpy()
        label_text = prompts[label]
        score_text = f"{score:.2f}"
        cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"{label_text} {score_text}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # Convert back to RGB for Gradio
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", sources="webcam")
            prompt_box = gr.Textbox(label="Detection Prompt (comma-separated)", value="a person")
        with gr.Column():
            output_img = gr.Image(label="Output")
        def process_stream(frame, prompt):
            return detect_objects(frame, prompt)
        input_img.stream(process_stream, [input_img, prompt_box], output_img, time_limit=15, stream_every=0.1, concurrency_limit=30)

if __name__ == "__main__":
    demo.launch()

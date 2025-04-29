import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers import pipeline
from ultralytics import YOLO

# Step 1: Define YOLO-specific configurations
class YOLOConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "YOLO"

# Step 2: Define a YOLO wrapper as a Hugging Face PreTrainedModel
class YOLOModel(PreTrainedModel):
    def __init__(self, config, model_path="./Object_Detection/siou150.pt"):
        super().__init__(config)
        
        # Load the YOLO model
        self.model = YOLO(model_path)

    def forward(self, images):
        # Perform inference
        return self.model(images)

# Step 3: Define a custom pipeline for YOLO
class YOLOPipeline:
    def __init__(self, model):
        self.model = model

    def __call__(self, image_paths):
        results = []
        for image_path in image_paths:
            # Run inference on each image
            result = self.model(image_path)
            results.append(result)
        return results

# Step 4: Load YOLO model as a Hugging Face-compatible model
def load_yolo_model(model_path):
    config = YOLOConfig()
    yolo_model = YOLOModel(config, model_path=model_path)
    return yolo_model

# Step 5: Main execution logic
if __name__ == "__main__":
    # Path to your local YOLO model
    local_yolo_model_path = "./Object_Detection/siou150.pt"

    # Load the YOLO model (use 'cpu' or 'cuda')
    device = "cuda"  # Change to 'cuda' for GPU if available
    yolo_model = load_yolo_model(local_yolo_model_path)

    # Wrap the YOLO model in a custom pipeline
    yolo_pipeline = YOLOPipeline(yolo_model)

    # Example images for inference
    test_images = ["./temp/photo.jpg"]

    # Perform inference
    results = yolo_pipeline(test_images)

    # Print and save the results
    for idx, result in enumerate(results):
        print(f"Results for {test_images[idx]}:")
        result.print()  # Print detection results
        result.save()   # Save annotated images in the same directory

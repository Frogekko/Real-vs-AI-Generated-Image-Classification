# Grad-CAM
# This script implements Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of an image that contribute most to the model's decision.
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import timm

model_path = "classifier_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model(model_path):
    model = 'convnextv2_tiny' # Explicitly defining the model architecture name
    model = timm.create_model(model, pretrained=True, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# This class is used to compute the Grad-CAM heatmap for a given model and target layer
class GradCAM:
    # Purposes of this function is to initialize the class and the model and the specific convolutional layer from which to extract features and gradients
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hooks are used by pytorch to intercept the data flowing through a layer during forward and backward passes
        target_layer.register_forward_hook(self.save_activation) # Saves activations to feature maps
        target_layer.register_full_backward_hook(self.save_gradient) # Saves gradients to target class

    # Stores the output of the target_layer during the forward pass
    def save_activation(self, module, input, output):
        self.activations = output.detach() # We uses detach() to avoid tracking gradients for memory efficiency

    # Stores the gradient of the target layer during the backward pass
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach() # Only the first entry within grad_output[0] are used, beecause thats the relevant gradient of the output with respect to the loss

    # This function will help to visualize  where the model is looking to make its decision
    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor) # Forward pass is our output
        if class_idx is None:
            class_idx = output.argmax(dim=1).item() # argmax is to choose the class with the highest output score, if its not specified

        self.model.zero_grad()
        output[0, class_idx].backward() # .backward() is called to populate gradients for the selected class

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # Global average pooling is used to calculate the mean gradient for each feature map channel
        current_activations = self.activations.to(pooled_gradients.device) # Ensure activations are on the same device as gradients

        for i in range(current_activations.shape[1]):
            current_activations[0, i, :, :] *= pooled_gradients[i]

        heatmap = current_activations[0].mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0: # Avoid division by zero if heatmap is all zeros
            heatmap /= np.max(heatmap)
        # If heatmap is all zeros, it remains all zeros, which is fine.
        return heatmap

# Overlay heatmap on image
def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.width, img.height)) # Resizes the heatmap to match the image dimensions
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # cv2 is used to turn grayscale heatmap into colored heatmap
    img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
    superimposed_img = heatmap_colored * 0.4 + img_np # Blends the heatmap with the image
    return cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB) # Converts the image back to RGB and ready to be displayed with matplotlib

# Main Grad-CAM runner
# Run_gradcam function executes the full grad-cam pipeline
def run_gradcam(img_path, model_path="classifier_model.pth"):
    # Define normalization transform only
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]) # Converts image to tensor and normalize it with ImageNet statistics

    # Load model
    model = load_model(model_path)

    # Specifying target layer for ConvNeXtV2 Tiny.
    # This path targets the depthwise convolution in the last block of the last stage.
    target_layer = model.stages[-1].blocks[-1].conv_dw
    
    cam = GradCAM(model, target_layer) # Using the GradCam class to get the heatmap

    # Open image
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Generate heatmap and overlay
    heatmap = cam.generate_heatmap(input_tensor)
    result = overlay_heatmap(img, heatmap)

    # Displayes the heatmap over the image
    plt.imshow(result)
    plt.axis('off')
    plt.title("Grad-CAM Visualization of attention")
    plt.show()

# Promps the user to choose the folder and image number to check the grad-cam
if __name__ == "__main__":
    file_name = input("Which folder do you want to test, real or fake? (real/fake) ").lower()
    image_nb = input("Which image do you want to check? (For example 0001, 0002, etc) ")
    image_type = ('.jpg', '.jpeg', '.png')
    base_path = f"~/Documents/03 - University/03 - Datasets/Real-vs-AI-Generated-Image-Classification/resized_test/resized_{file_name}"  # Change this to fit your folder path
    file_path = os.path.expanduser(base_path)

    found = False
    for ext in image_type:
        test_img_path = os.path.join(file_path, f"{image_nb}{ext}")
        if os.path.isfile(test_img_path):
            found = True
            break

    if found:
        run_gradcam(test_img_path)
    else:
        print(f"No file found for image {image_nb} in {file_path} with any of the extensions: {image_type}")
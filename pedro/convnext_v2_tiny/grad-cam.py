# Grad-CAM
# gradcam_visualizer.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

model_path = "classifier_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model(model_path):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Grad-CAM implementation
# 
class GradCAM:
    # Purposes of this function is to initialize the class and the model and the specific convolutional layer from which to extract features and gradients
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hooks are used by pytorch to intercept the data flowing through a layer during forward and backward passes
        target_layer.register_forward_hook(self.save_activation) # Saves activations to feature maps
        target_layer.register_backward_hook(self.save_gradient) # Saves gradients to target class

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
        for i in range(self.activations.shape[1]):
            self.activations[0, i, :, :] *= pooled_gradients[i]

        heatmap = self.activations[0].mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
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
# run_gradcam function executes the full grad-cam pipeline
def run_gradcam(img_path, model_path="classifier_model.pth"):
    # Define normalization transform only
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]) # Converts image to tensor and normalize it with ImageNet statistics, this is done because ResNet-18 is pretrained on ImageNet

    # Load model
    model = load_model(model_path)
    target_layer = model.layer4[1].conv2 # Specifying target layer, we choose a layer that is deep enough to capture high level semantics, yet still retains spatial dimensions
    cam = GradCAM(model, target_layer) # Using the GradCam class to get the heatmap

    # Open image
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # Generate heatmap and overlay
    heatmap = cam.generate_heatmap(input_tensor)
    result = overlay_heatmap(img, heatmap)

    # Displayes the heatmap over the image
    plt.imshow(result)
    plt.axis('off')
    plt.title("Grad-CAM Visualization of attention")
    plt.show()

# Promps the user to choose the folder and image number to check the 
if __name__ == "__main__":
    file_name = input("Whiche folder do you whant to test, real or fake? (real/fake) ").lower()
    image_nb = input("Which image do you whant to check? (int) ")
    image_type = ('.jpg', '.jpeg', '.png')
    file_path = f"F:/old_repo/Resized/resized_test/resized_test/resized_{file_name}"  # Change this to your fitting folder path

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
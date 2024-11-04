import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations (should match those used during validation)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the saved model weights and class names
checkpoint = torch.load('chest_xray_t1_model.pth', map_location=device)

# Retrieve class names
class_names = checkpoint['class_names']

# Reconstruct the model architecture
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Function to predict disease from a new image
def predict_image(image_path, model, data_transforms, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = data_transforms(img)
    batch_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(batch_t)
        _, preds = torch.max(outputs, 1)
        class_idx = preds.item()
        class_name = class_names[class_idx]

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f'Predicted: {class_name}')
    plt.axis('off')  # Hide axis
    plt.show()

# Test the model with new images
test_images = ['data/chest_xray/test/PNEUMONIA/person1_virus_8.jpeg',
                   'data/chest_xray/test/NORMAL/IM-0007-0001.jpeg']
for img_path in test_images:
    if os.path.exists(img_path):
        predict_image(img_path, model, data_transforms, class_names)
    else:
        print(f"Image not found: {img_path}")

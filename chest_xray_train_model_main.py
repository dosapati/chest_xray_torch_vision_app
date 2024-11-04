import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformations for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Training function
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} '
                  f'Acc: {epoch_acc:.4f}')
        print()

    return model

# Function to predict disease from a new image
def predict_image(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = data_transforms['val']
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        class_idx = preds.item()
        class_name = class_names[class_idx]

    # Display the image and prediction
    plt.imshow(Image.open(image_path), cmap='gray')
    plt.title(f'Predicted: {class_name}')
    plt.show()

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print('Confusion Matrix:')
    print(cm)

    return precision, recall, f1, cm


if __name__ == '__main__':
    # Data directories
    data_dir = 'data/chest_xray'

    # Load datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # Data loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=16,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['train', 'val']}

    # Dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Load a pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Modify the final layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5)
    #torch.save(model.state_dict(), 'trained_model.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, 'chest_xray_t1_model.pth')

    # Evaluate the model on the validation set
    print('Evaluating model on validation set...')
    precision, recall, f1, cm = evaluate_model(model, dataloaders['val'], class_names)

    # Test the model with two new images
    test_images = ['data/chest_xray/test/PNEUMONIA/person1_virus_8.jpeg',
                   'data/chest_xray/test/NORMAL/IM-0007-0001.jpeg']
    for img_path in test_images:
        predict_image(img_path, model, class_names)

# Version Modelo E 3 capas con DROPOUT (0,25 x 2), BS = 8, LR 10-5 y 100 Epochs

# Importamos librerias necesarias

from torchvision import transforms, datasets
import torchsummary
import torchvision
import torch
import time
from matplotlib import pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report

# Program beginning
if __name__ == '__main__':

    # Fijamos la semilla para reproducibilidad
    torch.manual_seed(12345)

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a transform function for Skin Cancer dataset
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset from the local directory
    #train_dir = 'C:/Users/gtoma/AI_Development/UEM_Master_AI_07042025/UEM_Trabajo/Sesiones_UEM/Manuel_Garcia_VISION/Actividad_1_Transfer_Learning/Nuevo_01052025/train'
    #test_dir = 'C:/Users/gtoma/AI_Development/UEM_Master_AI_07042025/UEM_Trabajo/Sesiones_UEM/Manuel_Garcia_VISION/Actividad_1_Transfer_Learning/Nuevo_01052025/test'
    train_dir = '/home/224F8578gianfranco/LORCA/train/'
    test_dir = '/home/224F8578gianfranco/LORCA/test/'
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
   
    print()
    print(f"Number of training samples: {len(train_dataset)}")
    train_dataset
    print(f"Number of testing samples: {len(test_dataset)}")
    test_dataset
    print()
    
    # Vamos a definir el tamaño del batch y el número de workers
    batch_size = 8
    num_workers = 4
    
    # Create data loaders for training and testing datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # Check the number of batches in the training and testing datasets
    print(f"Number of batches in training dataset: {len(train_loader)}")
    print(f"Number of batches in testing dataset: {len(test_loader)}")
    
    # Check the size of each batch in the training and testing datasets
    for images, labels in train_loader:
        print(f"Batch size in train: {images.size()}")
        break
    # Check the size of each batch in the testing dataset
    for images, labels in test_loader:
        print(f"Batch size in test: {images.size()}")
        break
    # Check the number of classes in the training and testing datasets
    print(f"Number of classes in training dataset: {len(train_dataset.classes)}")
    print(f"Number of classes in testing dataset: {len(test_dataset.classes)}")

    # Vamos a cargar el modelo preentrenado de ResNet18

    # Load the pre-trained ResNet18 model
    #model = models.resnet18(pretrained=True)
    model = models.resnet18(weights='DEFAULT')
    num_features = model.fc.in_features
    print()
    print(f"Number of features in the last fully connected layer: {num_features}")
    print()

    # Modify the last fully connected layer to match the number of classes in the dataset
    # 2 fully connected hidden layers and a binay classification    

    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(512, len(train_dataset.classes)),
        nn.Softmax(dim=1)
    )

    #model.fc = nn.Linear(num_features, len(train_dataset.classes))

    # Move the model to the GPU if available
    model = model.to(device)

    # Print the model architecture
    print(model)

    # Freeze the convolutional layers to prevent them from being updated during training
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define the loss function and optimizer.
    # Use CrossEntropyLoss for multi-class classification
    # Use Adam optimizer with a learning rate of 0.00001 (and L2 regularization)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

    # Print the optimizer and learning rate and device being used
    print()
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Using device: {device}")

    # Define the number of epochs for training
    num_epochs = 100

    # Let's get a model summary
    print()
    print("Model summary:")
    print()
    torchsummary.summary(model, (3, 224, 224), device=device.type)
    print()

    # Initialize lists to store training and testing loss and accuracy
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # Vamos a entrenar el modelo
    start_time = time.time()
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        running_loss = 0.0
        running_correct = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs.data, 1)  # Get the predicted class
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() # Accumulate loss
            running_correct += torch.sum(preds == labels.data).item() # Accumulate correct predictions 

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch
        epoch_acc = running_correct / len(train_loader.dataset)  # Average accuracy for the epoch

        train_loss_list.append(epoch_loss)  # Append loss to the list
        train_acc_list.append(epoch_acc)  # Append accuracy to the list

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
        # Evaluate the model on the test dataset
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs.data, 1)  # Get the predicted class
                loss = criterion(outputs, labels)  # Compute loss
                running_loss += loss.item() # Accumulate loss
                running_correct += torch.sum(preds == labels.data).item() # Accumulate correct predictions

            epoch_loss = running_loss / len(test_loader.dataset)  # Average loss for the epoch
            epoch_acc = running_correct / len(test_loader.dataset)  # Average accuracy for the epoch

            test_loss_list.append(epoch_loss)  # Append loss to the list
            test_acc_list.append(epoch_acc)  # Append accuracy to the list
            print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")

    end_time = time.time()
    print("Training completed.")
    print()
    print(f"Training completed in {num_epochs} epochs")
    print(f"Final training loss: {train_loss_list[-1]:.4f}")
    print(f"Final training accuracy: {train_acc_list[-1]:.4f}")
    print()
    print(f"Training started at: {time.ctime(start_time)}")
    print(f"Training ended at: {time.ctime(end_time)}")
    print(f"Training time: {end_time - start_time:.2f} seconds")

    # Vamos a guardar el modelo entrenado
    path_res = 'C:/Users/gtoma/AI_Development/UEM_Master_AI_07042025/UEM_Trabajo/Sesiones_UEM/Manuel_Garcia_VISION/Actividad_1_Transfer_Learning/Nuevo_01052025/'

    # Vamos a guardar el modelo entrenado
    torch.save(model.state_dict(), path_res + '/resnet18_skin_cancer_model_F.pth')
    
    # Vamos a graficar loss vs epochs train y test
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss vs Epochs')
    plt.grid()
    plt.show();

    # Vamos a guardar la grafica de loss vs epochs train y test
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss vs Epochs')
    plt.grid()
    plt.savefig(path_res + '/train_test_loss_vs_epochs_F.png')
    plt.close()  # Close the plot to free up memory

    # Vamos a graficar accuracy vs epochs train y test
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy vs Epochs')
    plt.grid()
    plt.show()
    
    # Vamos a guardar la grafica de accuracy vs epochs train y test
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy vs Epochs')
    plt.grid()
    plt.savefig(path_res + '/train_test_accuracy_vs_epochs_F.png')
    plt.close()  # Close the plot to free up memory
    
    # Vamos a graficar la matriz de confusion
    
    # Get the model predictions on the test set
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=train_dataset.classes, columns=train_dataset.classes)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Save the confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(path_res + '/confusion_matrix_F.png')
    plt.close()  # Close the plot to free up memory

    # Vamos a calcular la accuracy, la precision, el recall y el f1-score
    # Calculate the classification report
    report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    # Display the classification report
    print(report_df)
    
    # Save the classification report to a text file
    with open(path_res + '/classification_report_F.txt', 'w') as f:
        f.write(report_df.to_string())

    # Save the classification report to a CSV file
    report_df.to_csv(path_res + '/classification_report_F.csv', index=True)

    # Save the classification report to a JSON file
    report_df.to_json(path_res + '/classification_report_F.json', orient='index')
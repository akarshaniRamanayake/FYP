from matplotlib import pyplot as plt, gridspec
from torch import nn

from models import ANN
from lossFunctions import HuberLoss, CosineSimilarityLoss, LogCoshLoss

import torch.optim as optim
import torch

count = 0

def Trainer(GUI,train_loader, test_loader):
    global count

    fig1 = plt.figure(figsize=(5, 5))
    #gs1 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Define training parameters
    learning_rate = 0.001
    epochs = 100

    model = ANN()

    # loss functions
    # criterion = nn.MSELoss()
    # criterion = HuberLoss(delta=1.0)
    criterion = CosineSimilarityLoss()
    # criterion = LogCoshLoss()

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses = []
    validation_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            if (GUI.ifBreak):
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs + 1}, Loss: {epoch_loss}")

        # Evaluate validation loss
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                validation_loss += criterion(outputs, targets).item()
        validation_loss /= len(test_loader)
        validation_losses.append(validation_loss)

    #Plot the training curve
    ax1 = fig1.add_subplot(111)
    ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, epochs + 1), validation_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curve')
    ax1.legend()
    ax1.set_xlim(1, epochs)

    model.eval()
    test_loss = 0
    sdf = []
    confidence = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            #print(outputs)
            for i in outputs:
                sdf.append(i[0])
                confidence.append(i[1])
            test_loss += criterion(outputs, targets).item()
    test_loss /= len(test_loader)
    print("Test Loss:", test_loss)

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(sdf, confidence, c=['black']*len(sdf), s=1)
    ax2.set_xlabel('SDF')
    ax2.set_ylabel('Confidence')
    ax2.set_title('SDF vs Confidence')

    plt.tight_layout()
    plt.show()
    if count == 0:
        fig = ax1.get_figure()
        fig1.savefig('training_curve.png')
        # figure2 = ax2.get_figure()
        fig2.savefig('SDFvsConfidence.png')
        count = count + 1

    return model
    # print("obstacles",obstacle_points)

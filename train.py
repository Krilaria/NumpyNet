import matplotlib.pyplot as plt
from dataloader import My_Dataloader, BCE
from model import FCNet, CNNet
import numpy as np

train_dir = f'D:\Documentation_Python\Pet-projects\CatDogCLS\dataset128\\train'
valid_dir = f'D:\Documentation_Python\Pet-projects\CatDogCLS\dataset128\\valid'
weights_dir = f'D:\Documentation_Python\Pet-projects\CatDogCLS\weights.txt'
batch_size = 2
num_epochs = 6
hidden_dim = 32
output_size = 1
img_size = 32
learning_rate = 0.01

def trainFC(train_dir, valid_dir, weights_dir, batch_size, num_epochs, hidden_dim, output_size, img_size, learning_rate):
    train_loader = My_Dataloader(train_dir, batch_size, img_size=img_size)
    valid_loader = My_Dataloader(valid_dir, batch_size, shuffle=False, img_size=img_size)
    model = FCNet(img_size*img_size*3, hidden_dim, output_size)
    creterion = BCE
    weights = model.weights()
    with open(weights_dir, "w") as f:
        f.write(f"Model witn {round((model.num_par()/10**6), 1)}M parametems \n")
        for i in range(len(weights)):
            f.write(f"{weights[i]} \n")

    train_loss_history = []
    valid_loss_history = []
    accuraсy_history= []
    for epoch in range(num_epochs):
        print('________')
        print(f'Epoch: {epoch+1}/{num_epochs}')
        #train
        train_loss = 0
        learning_rate *= 0.9
        for batch_images, batch_labels in train_loader:
            X = batch_images.reshape(batch_images.shape[0], -1)
            y = batch_labels.reshape(-1, 1)
            output = model.forward(X)
            loss = creterion(output, y)
            model.backward(X, y, output, learning_rate)
            train_loss += loss.item()
            #print(model.W1[0][0])

        #validation
        valid_loss = 0
        correct = 0
        total = 0
        for batch_images, batch_labels in valid_loader:
            X = batch_images.reshape(batch_images.shape[0], -1)
            y = batch_labels.reshape(-1, 1)
            output = model.forward(X)
            loss = creterion(output, y)
            valid_loss += loss.item()
            prediction = np.where(output < 0.5, 0, 1)
            total += y.shape[0]
            correct += (prediction == y).sum().item()

        avg_train_loss = round(train_loss/train_loader.leng(), 3)
        avg_valid_loss = round(valid_loss/valid_loader.leng(), 3)
        accuraсy = round(correct / total * 100, 3)

        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(avg_valid_loss)
        accuraсy_history.append(accuraсy)

        print(f'avg_train_loss = {avg_train_loss}')
        print(f'avg_valid_loss = {avg_valid_loss}')
        print(f'accuraсy = {accuraсy}')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # График потерь
    ax1.plot(range(num_epochs), train_loss_history, color='red', label='Train Loss')
    ax1.plot(range(num_epochs), valid_loss_history, color='blue', label='Validation Loss')
    ax1.set_title("Loss Over Epochs", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(linestyle='--', alpha=0.7)

    # График точности
    ax2.plot(range(num_epochs), accuraсy_history, color='green', label='Validation Accuracy')
    ax2.set_title("Accuracy Over Epochs", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(linestyle='--', alpha=0.7)

    fig.suptitle("Training and Validation Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#trainFC(train_dir, valid_dir, weights_dir, batch_size, num_epochs, hidden_dim, output_size, img_size, learning_rate)

def trainCNN(train_dir, valid_dir, weights_dir, batch_size, num_epochs, output_size, img_size, learning_rate):
    kernel_size = 3
    hidden_size1 = 32*32*4
    hidden_size2 = 64
    train_loader = My_Dataloader(train_dir, batch_size, img_size=img_size)
    valid_loader = My_Dataloader(valid_dir, batch_size, shuffle=False, img_size=img_size)
    model = CNNet(img_size, kernel_size, hidden_size1, hidden_size2, output_size)
    creterion = BCE
    weights = model.weights()
    with open(weights_dir, "w") as f:
        f.write(f"Model witn {round((model.num_par()/10**6), 1)}M parametems \n")
        for i in range(len(weights)):
            f.write(f"{weights[i]} \n")

    train_loss_history = []
    valid_loss_history = []
    accuraсy_history= []
    for epoch in range(num_epochs):
        print('________')
        print(f'Epoch: {epoch+1}/{num_epochs}')
        #train
        train_loss = 0
        learning_rate *= 0.9
        for batch_images, batch_labels in train_loader:
            y = batch_labels.reshape(-1, 1)
            batch_images = batch_images.transpose(0, 3, 1, 2)
            output = model.forward(batch_images)
            loss = creterion(output, y)
            #print('output', output.shape)
            #exit()
            model.backward(batch_images, y, output, learning_rate)
            train_loss += loss.item()
            #print(model.W1[0][0])

        #validation
        valid_loss = 0
        correct = 0
        total = 0
        for batch_images, batch_labels in valid_loader:
            y = batch_labels.reshape(-1, 1)
            batch_images = batch_images.transpose(0, 3, 1, 2)
            output = model.forward(batch_images)
            loss = creterion(output, y)
            valid_loss += loss.item()
            prediction = np.where(output < 0.5, 0, 1)
            total += y.shape[0]
            correct += (prediction == y).sum().item()

        avg_train_loss = round(train_loss/train_loader.leng(), 3)
        avg_valid_loss = round(valid_loss/valid_loader.leng(), 3)
        accuraсy = round(correct / total * 100, 3)

        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(avg_valid_loss)
        accuraсy_history.append(accuraсy)

        print(f'avg_train_loss = {avg_train_loss}')
        print(f'avg_valid_loss = {avg_valid_loss}')
        print(f'accuraсy = {accuraсy}')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # График потерь
    ax1.plot(range(num_epochs), train_loss_history, color='red', label='Train Loss')
    ax1.plot(range(num_epochs), valid_loss_history, color='blue', label='Validation Loss')
    ax1.set_title("Loss Over Epochs", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(linestyle='--', alpha=0.7)

    # График точности
    ax2.plot(range(num_epochs), accuraсy_history, color='green', label='Validation Accuracy')
    ax2.set_title("Accuracy Over Epochs", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(linestyle='--', alpha=0.7)

    fig.suptitle("Training and Validation Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

trainCNN(train_dir, valid_dir, weights_dir, batch_size, num_epochs, output_size, img_size, learning_rate)
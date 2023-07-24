# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # 10 classes represent 10 probabilities for all ten objects.
        )

    def forward(self, x):  # output = model(input_data)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Training loop
def train(epochs,train_dataloader,test_dataloader, model, loss_fn, optimizer):

    best_acc = 0.
    train_L = []
    test_L = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        num_batches = len(train_dataloader)
        model.train()
        train_loss = 0.
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        mean_loss =  train_loss/num_batches
        print('Training Loss: ',mean_loss)
        train_L.append(mean_loss)

        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        mean_loss = test_loss/num_batches
        test_L.append(mean_loss)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        if best_acc < correct:
            best_acc = correct
            torch.save(model.state_dict(), save_path + "best_model.pth")
            print("Saved PyTorch Model State to model.pth")

def main():

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        img = X[0].transpose(0,2).numpy()
        img = np.dstack((img,img,img))

        plt.imshow(img)
        plt.show()
        print(y)
        break

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train(epochs, train_dataloader, test_dataloader, model, loss_fn, optimizer)

if __name__ == "__main__":

    save_path = 'data/'
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    main()

    # Use the best model to predict the labels of the images in test dataset.
    # test()

    # Tips: Loading models:
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load(save_path + "best_model.pth"))

    # visualization: pip install matplotlib
    # 2 curve training and testing loss
    # images


    # Try to improve the model accuracy:

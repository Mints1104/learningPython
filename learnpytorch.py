import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        # --- Define the Layers (The 'hardware' of our brain) ---
        
        # A linear layer that takes 10 input features and outputs 5.
        # This is our 'hidden layer'.
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        
        # A linear layer that takes the 5 features from layer1 and outputs 1 final prediction.
        # This is our 'output layer'.
        self.layer2 = nn.Linear(in_features=5, out_features=1)
        
    # The 'forward' method defines the path data takes through the brain.
    def forward(self, x):
        
        # 1. Pass the input 'x' through the first layer.
        x = self.layer1(x)
        
        # 2. Apply the ReLU activation function (the 'on/off' switch).
        x = nn.functional.relu(x)
        
        # 3. Pass the result through the second layer to get the final output.
        x = self.layer2(x)
        
        return x
    
X_train = torch.randn(100,10)
y_train = torch.randn(100,1)
net = SimpleNet()
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.01)

for epoch in range(50):
    #1. Forward Pass: Get the model's prediction
    prediction = net(X_train)
    #2. Calculate the Loss: How wrong was the prediction:
    loss = loss_function(prediction,y_train)
    #3. Zero Gradients: Reset the optimizer before calculaitng adjustments
    optimizer.zero_grad()
    #4. Backwards Pass (Backpropagation): Calculate how much each parameter contributed
    loss.backward()
    #5. Step: Update the model's parameters based on the calculations
    optimizer.step()
    # Print the loss every 10 epochs to see the progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
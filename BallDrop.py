import torch
import torch.nn as nn
import torch.nn.functional as F


class BallDrop(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size,out_size)

    def forward(self, input_vector):
        # Get intermediate outputs using hidden layer
        out = self.linear1(input_vector)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = F.relu(out)
        out = self.linear7(out)
        return out

    def training_step(self, inputs, outputs):
        loss_fn = F.mse_loss # Uses mean square error loss
        out = self(inputs)  # Generate predictions
        loss = loss_fn(out, outputs) # Calculate loss
        return loss

    # def validation_step(self, batch):
    #     images, labels = batch
    #     out = self(images)  # Generate predictions
    #     loss = F.cross_entropy(out, labels)  # Calculate loss
    #     acc = accuracy(out, labels)  # Calculate accuracy
    #     return {'val_loss': loss, 'val_acc': acc}
    #
    # def validation_epoch_end(self, outputs):
    #     batch_losses = [x['val_loss'] for x in outputs]
    #     epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    #     batch_accs = [x['val_acc'] for x in outputs]
    #     epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    #     return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    #
    # def epoch_end(self, epoch, result):
    #     print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
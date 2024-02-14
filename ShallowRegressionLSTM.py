import tqdm
from torch import nn
import torch


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, dropout):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        if 'nan' in [str(float(i)) for i in x.flatten() if str(float(i)) == 'nan']:
            print("detected nan in ShallowRegressionLSTM.forward - printing inputs vector:")
            print(x)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

    def train_epoch(self, data_loader, loss_function, optimizer):
        num_batches = len(data_loader)
        total_loss = 0
        self.train()

        for i, (X, y) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), mininterval=10):
            output = self(X)
            loss = loss_function(output, y)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if str(total_loss) == 'nan':
                print("detected nan in ShallowRegressionLSTM.train_epoch")

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}")
        return self, avg_loss

    def test_model(self, data_loader, loss_function):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()
        with torch.no_grad():
            for X, y in data_loader:
                output = self(X)
                total_loss += loss_function(output, y).item()
        avg_loss = total_loss / num_batches
        print(f"\nTest loss: {avg_loss}")
        return avg_loss

    def predict(self, data_loader):
        output = torch.tensor([])
        self.eval()
        with torch.no_grad():
            for i, (X, _) in enumerate(data_loader):
                y_star = self(X)
                output = torch.cat((output, y_star), 0)

        return output

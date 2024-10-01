import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.io as sio

SAMPLE_DS_PATH = "mg_data/ITER_like_equilibrium_dataset_sample.mat" # sample dataset
FULL_DS_PATH = 'mg_data/ITER_like_equilibrium_dataset.mat' # full dataset
MODEL_SAVE_PATH = "mg_data/mg_planet.pth"

class PlaNetDataset(Dataset):
    def __init__(self, ds_path):
        ds = sio.loadmat(ds_path)
        self.rr_pix = ds["RR_pixels"] # radial position of pixels (64, 64)
        self.zz_pix = ds["ZZ_pixels"] # vertical position of pixels (64, 64)
        self.input_currs = ds["DB_coils_curr_test_ConvNet"] # input currents (n,14)
        self.psi = ds["DB_psi_pixel_test_ConvNet"] # magnetic flux (n, 64, 64)
        self.psi = self.psi.transpose(0, 2, 1).reshape(-1, 64*64) # transpose and flatten
        self.input_currs = torch.tensor(self.input_currs, dtype=torch.float32) # convert to tensor
        self.psi = torch.tensor(self.psi, dtype=torch.float32) # convert to tensor
    def __len__(self):
        return len(self.psi)
    def __getitem__(self, idx):
        return self.input_currs[idx], self.psi[idx]

class PlaNet(torch.nn.Module):
    def __init__(self):
        super(PlaNet, self).__init__()
        n = 8
        self.fc1 = torch.nn.Linear(14, n)
        self.fc2 = torch.nn.Linear(n, n)
        self.fc3 = torch.nn.Linear(n, 64*64)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    EPOCHS = 10
    train_ds, val_ds = PlaNetDataset(FULL_DS_PATH), PlaNetDataset(SAMPLE_DS_PATH)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False) 
    model = PlaNet() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss() # Mean Squared Error Loss
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        trainloss, evalloss = [], []
        for input_currs, psi in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            psi_pred = model(input_currs)
            loss = loss_fn(psi_pred, psi)
            loss.backward()
            optimizer.step()
            trainloss.append(loss.item())
        model.eval()
        with torch.no_grad():
            for input_currs, psi in val_dl:
                psi_pred = model(input_currs)
                loss = loss_fn(psi_pred, psi)
                evalloss.append(loss.item())
        if sum(evalloss)/len(evalloss) < best_loss:
            best_loss = sum(evalloss)/len(evalloss)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Ep:{epoch+1}: Train Loss: {sum(trainloss)/len(trainloss):.4f}, Eval Loss: {sum(evalloss)/len(evalloss):.4f}")
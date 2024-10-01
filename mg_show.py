import torch
import matplotlib.pyplot as plt
from mg_train import PlaNetDataset, PlaNet, SAMPLE_DS_PATH, FULL_DS_PATH, MODEL_SAVE_PATH

if __name__ == "__main__":
    model = PlaNet()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    ds = PlaNetDataset(SAMPLE_DS_PATH)
    input_currs, psi = ds[0]
    psi_pred = model(input_currs.unsqueeze(0))
    psi_pred = psi_pred.detach().numpy().reshape(64, 64)
    psi = psi.detach().numpy().reshape(64, 64)
    err = ( (psi - psi_pred) / psi ) * 100
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    ext = [ds.rr_pix.min(), ds.rr_pix.max(), ds.zz_pix.min(), ds.zz_pix.max()]
    rr, zz = ds.rr_pix, ds.zz_pix  # radial and vertical positions of pixels

    im0 = axs[0].imshow(psi, extent=ext)
    axs[0].set_title("Actual")
    axs[0].set_aspect('equal')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(psi_pred, extent=ext)
    axs[1].set_title("Predicted")
    axs[1].set_aspect('equal')
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err, extent=ext)
    axs[2].set_title("Error %")
    axs[2].set_aspect('equal')
    fig.colorbar(im2, ax=axs[2])

    c0 = axs[3].contour(rr, zz, psi, levels=15, cmap='viridis', linestyles='dashed')
    c1 = axs[3].contour(rr, zz, psi_pred, levels=10, cmap='viridis')
    axs[3].set_title("Contours")
    axs[3].set_aspect('equal')

    plt.show()
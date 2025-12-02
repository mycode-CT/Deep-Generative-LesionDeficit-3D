import os
import glob
import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import nibabel as nib
from model import ModelWrapper

import random
import math
import monai
from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged

########################################################################################
########################################################################################

# ---------------------------------------------
# Load image files and reference template
# ---------------------------------------------

datafolder = '/datafolder'
all_image = sorted(glob.glob(os.path.join(datafolder, '*.nii.gz')))
load_data0 = datafolder + '/first_subject.nii.gz' # could be any subject
img0 = nib.load(load_data0)

# ---------------------------------------------------------
# Initialize NIHSS labels and set experiment configuration
# ---------------------------------------------------------
Num_of_subj = 2000 # Assuming we have 2000 subjects
allNIHSS = np.zeros((Num_of_subj, 15)) # There are 15 NIHSS subscores per subject. For this test run, they are initialized to zeros.
all_Z_DIM = np.array([10, 20, 50, 70]) # Z_DIM = 50 — a latent dimension of ~50 generally provides good performance.
EPOCHS = 4000
batch_size = 10 # Batch size
INPUT_SIZE = 128 # Each input volume is a 128×128×128 voxel cube.

save_fold_main_big = '/results'
if os.path.isdir(save_fold_main_big):
    print("The Big directory already exist")
else:
    print("Given Big directory doesn't exist, make new one")
    os.mkdir(save_fold_main_big)

for zz in range(len(all_Z_DIM)):
    Z_DIM = all_Z_DIM[zz]
    print(Z_DIM)
    save_fold_main = save_fold_main_big + '/ep_' + str(EPOCHS) + '_Z_DIM' + str(Z_DIM)

    if os.path.isdir(save_fold_main):
        print("The directory already exist")
    else:
        print("Given directory doesn't exist, make new one")
        os.mkdir(save_fold_main)

    # ---------------------------------------------
    # Select NIHSS subscore for this experiment
    # ---------------------------------------------
    sel_col = 10 # Ataxia subscore is located in column 11 of NIHSS (index 10 in 0-based indexing).
    deficit_scores = allNIHSS[:, sel_col]
    deficit_scores = np.reshape(deficit_scores, [Num_of_subj, ])
    file_namee = 'vae_final.nii.gz'

    # ---------------------------------------------------------
    # Create output directories and load template brain image
    # ---------------------------------------------------------
    save_fold = save_fold_main + '/'
    if os.path.isdir(save_fold):
        print('The save folder already exists')
    else:
        os.mkdir(save_fold)
    master_nii = nib.load(load_data0) # The template volume (master_nii) is used for saving generated masks with correct affine information.

    
    # ---------------------------------------------------------
    # Prepare datasets: normalise scores, split data, and build dataloaders
    # ---------------------------------------------------------
    save_fold_train = save_fold + '/train'
    save_fold_val = save_fold + '/val'
    if os.path.isdir(save_fold_train):
        print('The train folder already exists')
    else:
        os.mkdir(save_fold_train)
    if os.path.isdir(save_fold_val):
        print('The save folder already exists')
    else:
        os.mkdir(save_fold_val)

    n_deficit_scores = (deficit_scores - deficit_scores.mean()) / deficit_scores.std()
    all_image = sorted(glob.glob(os.path.join(datafolder, '*.nii.gz')))
    data_dicts = [{'image': image_name, 'label': def_score} for image_name, def_score in zip(all_image, n_deficit_scores)]
    random.shuffle(data_dicts)
    frac_train = 0.9
    frac_val = 0.05
    n_train = int(Num_of_subj*frac_train)
    n_val = int(Num_of_subj*frac_val)
    train_files = data_dicts[:n_train]
    val_files = data_dicts[n_train:n_train+n_val]
    cal_files = data_dicts[n_train+n_val:]
    all_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
        ]
    )
    train_ds = monai.data.Dataset(data=train_files, transform=all_transforms)
    train_loader_monai = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    val_ds = monai.data.Dataset(data=val_files, transform=all_transforms)
    val_loader_monai = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    cal_ds = monai.data.Dataset(data=cal_files, transform=all_transforms)
    cal_loader_monai = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # ---------------------------------------------------------
    # Initialize model, optimizer, and training configuration
    # ---------------------------------------------------------

    device = torch.device("cuda")
    CONTINUOUS = True
    INITIAL_CONV_KERNELS = 16
    L2_REG = 1e-5
    LR = 1e-4

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    Tensor = torch.cuda.FloatTensor

    model = ModelWrapper(INPUT_SIZE,
                         z_dim=Z_DIM,
                         start_dims=INITIAL_CONV_KERNELS,
                         continuous=CONTINUOUS).to(device)

    # Other optimisers work as well, Adamax is quite stable though
    optimizer = optim.Adamax(model.parameters(), weight_decay=L2_REG, lr=LR)

    print('NUM PARAMS: {}'.format(count_parameters(model)))
    print(f'NUM EPOCHS: {EPOCHS}')

    best_loss = 1e30
    best_acc = 0
    best_lk = 1e30
    global_step = 0

    # ---------------------------------------------------------
    # Main training loop with validation and performance logging
    # ---------------------------------------------------------

    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        model.zero_grad()
        train_acc = 0
        t_epoch_loss = 0

        # The trackers for the mean and scale of the inference map
        vae_mask = np.zeros((INPUT_SIZE, INPUT_SIZE, INPUT_SIZE))
        vae_scale = np.zeros((INPUT_SIZE, INPUT_SIZE, INPUT_SIZE))

        for batch_data in train_loader_monai:
            optimizer.zero_grad()
            x, y = batch_data["image"].type(Tensor).to(device), batch_data["label"].type(Tensor).to(device)
            y = torch.reshape(y, [torch.numel(y), 1])
            ret_dict, recon = model(x, y)
            loss = ret_dict['loss'].mean()
            t_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            vae_mask += np.squeeze(ret_dict['mean_mask'].cpu().data.numpy())
            vae_scale += np.squeeze(ret_dict['mask_scale'].cpu().data.numpy())
            train_acc += 1
            global_step += 1

        training_losses.append(t_epoch_loss / train_acc)

        vae_mask = vae_mask / train_acc
        val_mask = tc.from_numpy(vae_mask).type(Tensor).to(device).view(1, 1,
                                                                        INPUT_SIZE,
                                                                        INPUT_SIZE,
                                                                        INPUT_SIZE)
        vae_scale = vae_scale / train_acc
        val_scale = tc.from_numpy(vae_scale).type(Tensor).to(device).view(1, 1,
                                                                          INPUT_SIZE,
                                                                          INPUT_SIZE,
                                                                          INPUT_SIZE)

        val_acc = 0
        accuracy_acc = 0
        loss_acc = 0
        likelihood_acc = 0
        kld_acc = 0
        recon_acc = 0
        with torch.no_grad():
            for batch_data in val_loader_monai:
                x, y = batch_data["image"].type(Tensor).to(device), batch_data["label"].type(Tensor).to(device)
                y = torch.reshape(y, [torch.numel(y), 1])

                ret_dict, recon = model(x, y,
                                 provided_mask=val_mask,
                                 provided_scale=val_scale,
                                 val=True)

                loss_acc += ret_dict['loss'].mean().item()
                val_acc += 1
                likelihood_acc += ret_dict['mask_ll'].item()
                accuracy_acc += ret_dict['acc'].item()
                kld_acc += ret_dict['kl'].item()
                recon_acc += ret_dict['recon_ll'].item()

        loss = loss_acc / val_acc
        validation_losses.append(loss)
        lk = likelihood_acc / val_acc
        acc = round(accuracy_acc / val_acc, 4)
        kl = round(kld_acc / val_acc, 3)
        rec = recon_acc / val_acc

        print(f'Epoch: {epoch}, mask likelihood: {lk}, KL: {kl}, recon likelihood: {rec}')

        if lk < best_lk:
            best_loss = loss
            best_lk = lk
            best_acc = acc
            best_recon = recon_acc
            best_epoch = epoch

        if epoch % 10 == 0:
            print(f'Best: {best_lk}, {best_loss}, {best_acc}, epoch: {best_epoch}')

    executed_epochs = [i for i in range(len(training_losses))]
    train_log_loss = [math.log(l) for l in training_losses]
    val_log_loss = [math.log(l) for l in validation_losses]

    plt.plot(executed_epochs[10:], train_log_loss[10:])
    plt.plot(executed_epochs[10:], val_log_loss[10:])
    plt.show()

    np.save(os.path.join(save_fold, 'executed_epochs.npy'), executed_epochs)
    np.save(os.path.join(save_fold, 'train_log_loss.npy'), train_log_loss)
    np.save(os.path.join(save_fold, 'val_log_loss.npy'), val_log_loss)

    # ---------------------------------------------------------
    # Post-training mask CALIBRATION: persistence, boundary cleanup,
    # threshold optimisation, and NIfTI export
    # ---------------------------------------------------------
    torch.save(model, os.path.join(save_fold, 'vae.pth'))
    np.save(os.path.join(save_fold, 'vae_mask.npy'), vae_mask)

    model = torch.load(os.path.join(save_fold, 'vae.pth'))
    model.eval()
    vae_mask = np.load(os.path.join(save_fold, 'vae_mask.npy'))

    vae_mask[:, :2, :] = 0
    vae_mask[:2, :, :] = 0
    vae_mask[:, :, :2] = 0

    vae_mask[:, -2:, :] = 0
    vae_mask[-2:, :, :] = 0
    vae_mask[:, :, -2:] = 0

    master_nii = nib.load(load_data0)
    vae_mask_nii = nib.Nifti1Image(vae_mask, master_nii.affine)
    nib.save(vae_mask_nii, os.path.join(save_fold, file_namee))

    best_threshold = 0
    best_acc = 1e30

    threshold_range = np.linspace(0.905, 0.995, num=50)
    thrd_all_acc = []
    thrd_all_thrd = []
    for thresh in threshold_range:
        with torch.no_grad():
            counter = 0
            accuracy = 0
            for batch_data in cal_loader_monai:
                x, y = batch_data["image"].type(Tensor).to(device), batch_data["label"].type(Tensor).to(device)
                y = torch.reshape(y, [torch.numel(y), 1])
                ret_dict, recon = model(x, y,
                                 calibrate=True,
                                 t=float(thresh))
                accuracy += ret_dict['acc']
                counter += 1
            accuracy = accuracy / counter
            print(f'{thresh}, {accuracy}')

            if accuracy < best_acc:
                best_acc = accuracy
                best_threshold = thresh

    t = np.quantile(vae_mask, best_threshold)
    vae_mask = (vae_mask > t) * vae_mask

    vae_mask_nii_thrd = nib.Nifti1Image(vae_mask, master_nii.affine)
    nib.save(vae_mask_nii_thrd, os.path.join(save_fold, 'thrd_' + file_namee))

    del vae_mask, master_nii, t, vae_mask_nii, vae_mask_nii_thrd

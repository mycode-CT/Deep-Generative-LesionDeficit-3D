# Deep-Generative-LesionDeficit-3D
3D deep generative lesion–deficit mapping model used in Tangwiriyasakul et al., Deep generative computed perfusion-deficit mapping of ischaemic stroke (Communications Biology, accepted with minor revision)

Companion Code for:
Tangwiriyasakul et al., Deep generative computed perfusion-deficit mapping of ischaemic stroke, Communications Biology (year, to be announced).

Overview
This repository contains the custom 3D deep generative lesion–deficit mapping code used in the above manuscript.
The provided scripts implement a 3D extension of the variational lesion–deficit framework presented in:
Pombo et al., “Deep Variational Lesion-Deficit Mapping,” 2023
Original code: https://github.com/guilherme-pombo/vae_lesion_deficit

The modifications in this repository adapt the method to full 3D volumetric neuroimaging inputs (128 × 128 × 128) and implement the generative inference model, calibration steps, and training pipeline described in Tangwiriyasakul et al.

This repository includes only the custom code created for this study; it does not include any proprietary preprocessing code.

Contents
model.py      - 3D VAE-based lesion–deficit model (architecture, inference, KL, decoding)
train3D.py    - Example training, validation, and calibration pipeline

Requirements

This code is written for Python 3 and requires the following packages:
PyTorch (GPU recommended)
MONAI
NumPy
Nibabel
Matplotlib

Install with: pip install torch monai nibabel numpy matplotlib

Input Data Requirements
To run the training script, you must supply:
1) 3D lesion volumes
  Format: *.nii.gz
  Shape: 128 × 128 × 128
  Path: set via the datafolder variable in train3D.py
2) Deficit scores (e.g., NIHSS subscores)
   The script expects one score per subject
   Insert your own scores where indicated in train3D.py

How to Run
1) Place all .nii.gz volumes in a single folder (e.g., ./data/).
2) Edit train3D.py:
      Set datafolder = './data'
      Load your own deficit scores
3) Run: python train3D.py
  The script will:
  Train the 3D lesion–deficit model
  Perform validation
  Calibrate inference maps
  Save outputs (e.g., vae.pth, inference masks, log files) into a results folder

Citation

If you use this code, please cite:

Tangwiriyasakul et al., Deep generative computed perfusion-deficit mapping of ischaemic stroke, Communications Biology (year to be announced).
https://github.com/mycode-CT/Deep-Generative-LesionDeficit-3D

and

Pombo et al., Deep Variational Lesion-Deficit Mapping, 2023.
https://github.com/guilherme-pombo/vae_lesion_deficit

License
Specify whichever license you prefer (e.g., MIT, Apache 2.0).
MIT is commonly used for academic code.

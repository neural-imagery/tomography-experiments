# Tomography Experiments

The goal of tomography is to turn primitive fNIRS sensor measurements into a map of neural activity.

## Folder structure

* `3d` uses Monte Carlo photon transport to run tomography in 3D
* `recon_2obj` includes TOAST++ simulation code for a simple 2D reconstruction for both continuous wave (CW) and time-domain (TD) fNIRS systems.
* `fmri2fnirs` includes a pipeline for turning fMRI data into pseudo-fNIRS data and apply it to the Natural Scenes Dataset (NSD).

### Installation
```
pipenv install
```

For some reason, pipenv doesn't install jax nicely. So once you're in the environment (using `pipenv shell`), run
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# README

## Install tensorflow-gpu in venv
1. Navigate to the venv folder, e.g.:
`C:\Users\AStringfield\repositories\FashionMNIST\venv`
2. Run the pip commmand:
`pip pip install tensorflow-gpu`

#### *Note*
At the time of installing, the recommended tensorflow-gpu version based on the installed versions of CUDA and cuDNN libraries was tensorflow-gpu-2.6, which had previously been
used with success (installed in the base Python directory, not the new venv for this project).
However, when I ran the `pip` command listed above, this installed tensorflow-gpu-2.7. Even though the CUDA/cuDNN dependencies
hadn't been installed for this version, running the following test script indicated that
tensorflow was able to detect my graphics card with no problem:
```python
import tensorflow as tf
with tf.compat.v1.Session() as sess:
    devices = sess.list_devices()
devices
```

```text
2021-11-07 07:14:16.970792: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-07 07:14:17.482152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2154 MB memory:  -> device: 0, name: Quadro T1000 with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 7.5
```

This was then validated by running training on a test model in `tf_gpu_validation.py` and observing the GPU usage
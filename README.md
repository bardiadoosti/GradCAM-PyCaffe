# GradCAM-PyCaffe

[GradCAM][https://github.com/ramprs/grad-cam] implemented for PyCaffe

Based on http://sandaw89.blogspot.com/2017/08/gradcam-implementation-in-pycaffe.html

Make sure to first install [Caffe][https://github.com/BVLC/caffe] and set `caffe_root` at the top of the code.

To download pretrained model run the bash below after you installed caffe:

```
./scripts/download_model_binary.py models/bvlc_reference_caffenet
./data/ilsvrc12/get_ilsvrc_aux.sh
```

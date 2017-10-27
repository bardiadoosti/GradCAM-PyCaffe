import numpy as np
import sys
import os
caffe_root = '/path/to/caffe/'
sys.path.insert(1, caffe_root+'python/')
import caffe
import cv2

net_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net_model = caffe_root + 'models/bvlc_reference_caffenet/deploy2.prototxt'
image_address = caffe_root + 'examples/images/cat.jpg'

# load labels
categories = np.load('categories.npy')

# load the model
net = caffe.Net(net_model, net_weights, caffe.TEST)
#input_model = caffe.Net(net_model2, net_weights, caffe.TEST)

# load input and preprocess it
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# reshape the image while classifing only one image
net.blobs['data'].reshape(1,3,227,227)

# load the image to the data layer of the model
im = caffe.io.load_image(image_address)
net.blobs['data'].data[...] = transformer.preprocess('data', im)

# classify the image
out = net.forward()

#predicted class
#print(out['prob'].argmax())

final_layer = "fc8" #output layer whose gradients are being calculated
image_size = (227,227) #input image size
filter_shape = (13, 13) #size of the filter of 'conv5'
layer_name = 'conv5' #convolution layer of interest
category_index = out['prob'].argmax() #-if you want to get the saliency map of predicted class or else you can get saliency map for any interested class by specifying here

#Make the loss value class specific    
label = np.zeros(net.blobs[final_layer].shape)
label[0, category_index] = 1    

imdiff = net.backward(diffs= ['data', layer_name], **{net.outputs[0]: label}) 
gradients = imdiff[layer_name] #gradients of the loss value/ predicted class score w.r.t conv5 layer

#Normalizing gradients for better visualization
gradients = gradients/(np.sqrt(np.mean(np.square(gradients)))+1e-5)
gradients = gradients[0,:,:,:]

print("Gradients Calculated")

activations = net.blobs[layer_name].data[0, :, :, :] 

#Calculating importance of each activation map
weights = np.mean(gradients, axis=(1, 2))

cam = np.ones(filter_shape, dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * activations[i, :, :]    

#Let's visualize Grad-CAM
cam = cv2.resize(cam, image_size)
cam = np.maximum(cam, 0)
heatmap = cam / np.max(cam)
cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) 

#We are going to overlay the saliency map on the image
new_image = cv2.imread(image_address)
new_image = cv2.resize(new_image, image_size)

cam = np.float32(cam) + np.float32(new_image)
cam = 255 * cam / np.max(cam)
cam = np.uint8(cam)

#Finally saving the result
cv2.imwrite(categories[category_index] + ".jpg", cam)

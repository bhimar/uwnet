# We calculate the number of operations from the matrix multiplication in each convolutional layer. The activation layer and maxpool layers
# do not have matrix multiplication operations that we need to consider. Then we made a connected neural network with the same number of
# matrix multiplication operations.

# The accuracies for the conv_net were train: 0.700, test: 0.658


from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1), # 8 x 27 x 1024 = 221184
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1), # 16 x 72 x 256 = 294912
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1), # 32 x 144 x 64 = 294912
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1), # 64 x 288 x 16 = 294912
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10), # 1 x 256 x 10 = 2560
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def connected_net():
    l = [   make_connected_layer(3072, 72),
            make_activation_layer(RELU),
            make_connected_layer(72, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 1104),
            make_activation_layer(RELU),
            make_connected_layer(1104, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 10), # 1 x 256 x 10 = 2560
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = connected_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#


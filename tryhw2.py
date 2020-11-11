from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(4),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = 0.12
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? 
# How does it affect convergence? How does it affect what magnitude of learning rate you can use? 
# Write down any observations from your experiments:
# TODO: Your answer
# We compared the testing accuracies of the regular convnet against convnet 
# with batchnorm for the following learning rates:
# --------------------------
# lr	batch-norm	regular
# --------------------------
# 0.005	0.507	    0.332
# 0.01	0.527	    0.403
# 0.05	0.526	    0.483
# 0.07	0.531	    0.438
# 0.09	0.509	    0.447
# 0.1	0.502	    0.369
# 0.12	0.461	    0.1

# The convolutional network with batchnorm performed better than the regular 
# convolutional network. As you can observe from the table, the convolutional 
# network with batchorm delivers similar performance across different learning
# rates. But the regular convolutional network performed weakly for lower learning
# rates like 0.05 and higher learning rates like 0.12
# The optimal learning rate for convnet with batchnorm is with learning rate = 0.07
# and testing accuracy = 0.531. We can see that for a higher learning rate like 0.1,
# the testing accuracy is similar (0.502). This shows that we can train with faster
# convergence with batch-norm.

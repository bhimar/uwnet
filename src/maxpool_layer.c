#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    int out_cols =  outw*outh*l.channels;
    int img_size = l.width * l.height * l.channels; 
    matrix out = make_matrix(in.rows, out_cols);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    // for every image in the batch
    for(int img = 0; img < in.rows; img++) {
        // for every channel in an image
        int out_idx = 0;
        for (int k = 0; k < l.channels; k++) {
            // track center of pooling filter
            for(int i = 0; i < l.height; i+= l.stride) {
                for(int j = 0; j < l.width; j+= l.stride) {
                    
                    float max_el = FLT_MIN;
                    for (int m = i - (l.size / 2) + (1 - (l.size % 2)); m <= i + (l.size / 2); m++) {
                        for (int n = j - (l.size / 2) + (1 - (l.size % 2)); n <= j + (l.size / 2); n++) {
                            if (m >= 0 && n >= 0 && m < l.height && n < l.width) {
                              float el = in.data[img * img_size + k * l.width * l.height + m * l.width + n];
                              if (el > max_el) {
                                  max_el = el;
                              }
                            }
                        }
                    }
                    // populate out
                    out.data[img * out_cols + out_idx] = max_el;
                    out_idx++;
                }
            }
        }
    }


    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int img_size = dx.cols;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    int out_cols =  outw*outh*l.channels;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

     for(int img = 0; img < in.rows; img++) {
        // for every channel in an image
        int out_idx = 0;
        for (int k = 0; k < l.channels; k++) {
            // track center of pooling filter
            for(int i = 0; i < l.height; i+= l.stride) {
                for(int j = 0; j < l.width; j+= l.stride) {
                    
                    float max_el = FLT_MIN;
                    int max_ind = img * img_size + k * l.width * l.height + i * l.width + j; 

                    for (int m = i - (l.size / 2) + (1 - (l.size % 2)); m <= i + (l.size / 2); m++) {
                        for (int n = j - (l.size / 2) + (1 - (l.size % 2)); n <= j + (l.size / 2); n++) {

                            if (m >= 0 && n >= 0 && m < l.height && n < l.width) {
                              float el = in.data[img * img_size + k * l.width * l.height + m * l.width + n];
                              int ind = img * img_size + k * l.width * l.height + m * l.width + n;
                              if (el > max_el) {
                                  max_el = el;
                                  max_ind = ind;
                              }
                            }

                        }
                    }
                    // populate out
                    dx.data[max_ind] += dy.data[img * out_cols + out_idx];
                    out_idx++;
                }
            }
        }
    }


    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}


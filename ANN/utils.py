# Activate
#   |- Linear
#   |- ReLU
#   |- Sigmoid
#   |- Softmax
#   |- Tanh
# Regularize
#   |- L2
#   |- Dropout
# Initialize (variance)
#   |- Constant
#   |- He
#   |- Xavier
# Optimize
#   |- EMA
#   |- RMSprop
#   |- Adam
# Normalization/ standardization
# Grad checking

import numpy as np
import gc
import matplotlib.pyplot as plt


class Activate:
    """
    Following activation functions are defined under this class
    (1) Linear
    (2) ReLU (Rectified Linear Unit)
    (3) Sigmoid
    (4) Softmax
    (5) Tanh (Hyperbolic tangent)
    """

    def linear(self, x):
        """
        Linear activation function (pass-through).
        Returns nothing but the the input vector itself.
        """
        return x

    def relu(self, x, max_value=None, threshold=0):
        """
        Rectified linear unit.
        With default values, it returns element-wise `max(x, 0)`.
        Otherwise:
        `relu(x) = max_value` for `x >= max_value`,
        `relu(x) = x` for `threshold <= x < max_value`,

        Parameters
        ----------
        x : ndarray
        max_value: float, default = None
        threshold : float, default = 0
        """

        output = np.maximum(x, threshold)
        output = np.minimum(output, max_value)

        return output

    def sigmoid(self, x):
        """
        Sigmoid activation function,
            `sigmoid(x) = 1 / (1 + exp(-x))`.

        The sigmoid function always returns a value between 0 and 1.

        For example:
            x = [-20, -1.0, 0.0, 1.0, 20]
            output = [2.06e-09, 2.689e-01, 5.00e-01, 7.31e-01, 1.0]
        """
        output = 1 / (1 + np.exp(-x))
        return output

    def softmax(self, x, axis=-1):
        """
        Softmax converts a vector of values to a probability distribution. It can
        be viewed as a higher-dimensional generalization of the sigmoid function.

        Parameters
        ----------
        x : ndarray of shape ()
        axis : int, default=-1
            axis along which the softmax normalization is applied.

        Returns
        -------
        output : ndarray of shape ()
            Output of softmax transformation (all values must be non-negative
            and sum to 1).
        """

        norm = np.sum(np.exp(x))
        output = np.exp(x)/norm

        # assert all the values are non-negative
        assert all(output > 0)
        # All the elements of the output vector sum to 1.
        assert np.sum(output) == 1.0

        return output

    def tanh(self, x):
        """
        Hyperbolic tangent activation function.

        Returns:
            ndarray of same shape and dtype as those of input `x`.
        """
        return np.tanh(x)


# ----- Methods useful for Convolutional Neural Networks (CNN) ---- #
def conv_dims(dim: int, f: int, pad: int, stride: int):
    """
    Arguments:
        dim -- int; dim of the input image
        f -- int; dim of the filter/ weight (square) matrix
        pad -- int; padding level
        stride -- int; stride parameter

    Returns:
        dim -- dim of the output layer after convolution
    """

    dim = 1 + int((dim - f + 2 * pad) / stride)

    return dim


def zero_pad(image, pad: int, demo=False):
    """
    Pad the images with zeros

    Parameters:
        image -- ndarray; of shape (m, nH, nW, nc) representing a vector of m images
        pad -- int; amount of padding around height, width dimensions of each image
        demo -- bool; show the image pixels before and after padding

    Returns:
        im_pad -- padded image of shape (m, nH + 2*pad, nW + 2*pad, nc)

    Raises:
        ValueError -- Input dimension must be (m, nH, nW, nc)
    """

    # Check the input format
    if len(image.shape) < 4:
        raise ValueError("Input dimensions not of the form (m, nh, nw, nc).")

    im_pad = np.pad(image, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                    mode='constant', constant_values=(0, 0))

    # If 'demo' requested then show image before & after padding
    if demo:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Original Image')
        ax[0].imshow(image[0, :, :, 0])
        ax[1].set_title('Padded Image')
        ax[1].imshow(im_pad[0, :, :, 0])
        plt.show()

    return im_pad


def conv_block(im_block, kernel, bias):
    """
    Convolve the filter with a single block matrix of the same size.

    Parameters:
        im_block -- ndarray, slice of input image of shape (f, f, nC_prev)
        kernel -- ndarray, Weight parameters of the filter matrix of shape (f, f, nC_prev)
        bias -- float, Bias parameters - matrix of shape (1, 1, 1)

    Returns:
        conv_im -- float, result of convolution with (kernel, bias)
    """

    # Element-wise product between im_block and kernel.
    s = np.multiply(im_block, kernel)
    # Sum over all entries of the volume s and add the bias term.
    conv_im = np.sum(s) + float(bias)

    return conv_im


def conv_full(image, kernel, bias, stride, pad, demo=False):
    """
    Implements the convolution operation on the full input image. This is done
    by repeating 'conv_block' method on the entire image.

    Arguments:
        image -- output activations of the previous layer,
            ndarray of shape (m, nH_prev, nW_prev, nC_prev)
        kernel -- Weights, ndarray of shape (f, f, nC_prev, nC)
        bias -- Biases, ndarray of shape (1, 1, 1, nC)
        stride -- int; stride parameter
        pad -- int; padding parameter
        demo -- bool; show the image pixels before and after convolution

    Returns:
        im_out -- convolved output, numpy array of shape (m, nH, nW, nC)
    """

    # Retrieve the shapes of the matrices
    (m, nh_prev, nw_prev, nc_prev) = image.shape
    # nc = num channels in the output layer = num filters
    (f, f, nc_prev, nc) = kernel.shape

    # Compute the number of dims in height and width of the output
    nh = conv_dims(nh_prev, f, pad, stride)
    nw = conv_dims(nw_prev, f, pad, stride)

    # The convolved output image is initialized by zeros
    im_out = np.zeros(shape=(m, nh, nw, nc))

    # Pad the input image before convolution begins and overwrite
    image = zero_pad(image, pad, demo=False)

    for i in range(m):  # loop over the training examples
        im_i = image[i, :, :, :]

        for h in range(nh):  # loop over the vertical axis of the matrix
            top = h * stride
            bottom = top + f

            for w in range(nw):  # loop over the horizontal axis of the matrix
                left = w * stride
                right = left + f

                for c in range(nc):  # loop over the channels
                    im_block = im_i[top:bottom, left:right, :]
                    weights = kernel[:, :, :, c]
                    biases = bias[:, :, :, c]
                    # Convolve the image block
                    im_out[i, h, w, c] = conv_block(im_block, weights, biases)

    # Making sure the output shape is correct
    assert (im_out.shape == (m, nh, nw, nc))

    # If 'demo' requested then show image before & after convolution
    if demo:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Original Image (%ix%ix%i)'
                        % (nh_prev + 2 * pad, nw_prev + 2 * pad, nc_prev))
        ax[0].imshow(image[0, :, :, 0])
        ax[1].set_title('Convolved Image (%ix%ix%i)' % (nh, nw, nc))
        ax[1].imshow(im_out[0, :, :, 0])
        plt.show()

    # if input matrix is large enough it's good to deallocate memory
    del image
    gc.collect()

    return im_out


def pooling(image, f: int, stride: int, mode='max', demo=False):
    """
    Implements the convolution operation on the full input image. This is done
    by repeating 'conv_block' method on the entire image.

    Arguments:
        image -- output activations of the previous layer,
            ndarray of shape (m, nH_prev, nW_prev, nC_prev)
        f -- int; dim of the filter/ weight (square) matrix
        stride -- int; stride parameter
        mode -- 'max' for max pooling; 'avg' for average pooling
        demo -- bool; show the image pixels before and after pooling

    Returns:
        im_out -- pooled output, ndarray of shape (m, nH, nW, nC)

    Exceptions:
        KeyError -- raised if 'mode' key is incorrectly specified.
    """

    # Retrieve the shapes of the input
    (m, nh_prev, nw_prev, nc_prev) = image.shape

    # Compute the number of dims in height and width of the output
    nh = conv_dims(nh_prev, f, 0, stride)  # pooling doesn't need padding
    nw = conv_dims(nw_prev, f, 0, stride)
    nc = nc_prev

    # The convolved output image is initialized by zeros
    im_out = np.zeros(shape=(m, nh, nw, nc))

    for i in range(m):  # loop over the training examples
        im_i = image[i, :, :, :]

        for h in range(nh):  # loop over the vertical axis of the matrix
            top = h * stride
            bottom = top + f

            for w in range(nw):  # loop over the horizontal axis of the matrix
                left = w * stride
                right = left + f

                for c in range(nc):  # loop over the channels
                    im_block = im_i[top:bottom, left:right, c]

                    if mode == 'max':
                        im_out[i, h, w, c] = np.max(im_block)
                    elif mode == "avg":
                        im_out[i, h, w, c] = np.mean(im_block)
                    else:
                        raise KeyError("'mode' key unrecognized, use 'avg' or 'max'.")

    # Making sure the output shape is correct
    assert (im_out.shape == (m, nh, nw, nc))

    # If 'demo' requested then show image before & after convolution
    if demo:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Original Image')
        ax[0].imshow(image[0, :, :, 0])
        ax[1].set_title('%s-pooled Image' % mode)
        ax[1].imshow(im_out[0, :, :, 0])
        plt.show()

    del image
    gc.collect()

    return im_out

# def forward_prop(self, X):
#   CONV2D -> RELU -> MAX_POOL -> CONV2D -> RELU -> MAX_POOL -> FLATTEN -> FULLY_CONNECTED
#   CONV2D: stride of 1, padding 'SAME'
#     Z1 = None
#     # RELU
#     A1 = None
#     # MAX_POOL: window 8x8, stride 8, padding 'SAME'
#     P1 = None
#     # CONV2D: filters W2, stride 1, padding 'SAME'
#     Z2 = None
#     # RELU
#     A2 = None
#     # MAX_POOL: window 4x4, stride 4, padding 'SAME'
#     P2 = None
#     # FLATTEN
#     F = None
#     # FULLY-CONNECTED without non-linear activation function (not not call softmax).
#     # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
#     Z3 = None --> return Z3 (a linear unit) as a 'self' object.

# ----- Speeding methods for Artificial Neural Networks (ANN) ---- #

#
# def init_var_scale(num_hidden_units, key, const=0.01):
#     # Good for ReLU
#     if key == 'He':
#         # pass the number of hidden units in the previous layer
#         return np.sqrt(2. / num_hidden_units)
#
#     # Good for Sigmoid or Tanh
#     elif key == 'Xavier':
#         # pass the number of hidden units in the previous layer
#         return np.sqrt(1. / num_hidden_units)
#
#     # If not sure, just use a constant variance
#     else:
#         return const
#
#
# def accuracy(y_orig, y_pred):
#     return round(np.mean(y_orig == y_pred) * 100, 4)
#
#
# def activate(x, key):
#
#     if key == 'sigmoid':
#         a_func = 1 / (1 + np.exp(-x))
#         a_grad = np.multiply(a_func, 1-a_func)
#
#     elif key == 'tanh':
#         a_func = np.tanh(x)
#         a_grad = 1 - np.power(a_func, 2)
#
#     elif key == 'ReLU':
#         a_func = np.maximum(0, x)
#         a_grad = np.heaviside(x, 0)  # x==0 returns 0
#
#     else:
#         raise KeyError("Invalid activation key. "
#                        "Choose from 'tanh', 'sigmoid', 'ReLU'")
#
#     return a_func, a_grad
#
#
# def train_test(A, B, test_size=0.2):
#
#     nums = A.shape[1]
#     frac = round(nums * (1 - test_size))
#
#     # Shuffle the index array and then map that to X, Y
#     idx = np.arange(nums)
#     np.random.shuffle(idx)
#     A = A[:, idx]
#     B = B[:, idx]
#
#     return A[:, :frac], A[:, frac:], B[:, :frac], B[:, frac:]
#
#
# def set_bias_as_weight(shape):
#     return shape[0] + 1, shape[1]
#
#
# def add_bias(vector):
#     return np.hstack([vector, np.ones((vector.shape[0], 1))])
#
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
#
# def shuffle_vectors(x, y):
#     rd = np.arange(len(x))
#     np.random.shuffle(rd)
#     x = x[rd]
#     y = y[rd]
#     return x, y
#
#
# def _stable_clip(x):
#     """Used to avoid numerical inestability when"""
#     return np.clip(x, 1e-7, 1 - 1e-7)
#
#
# def mean_squared_error(ypred, ytrue):
#     return (ypred - ytrue) * ypred * (1 - ypred)
#
#
# def cross_entropy(ypred, ytrue, binary=True):
#     # return -ytrue * np.log(_stable_clip(ypred)) -\
#     #         (1 - ytrue) * np.log(1 - _stable_clip(ypred))
#     return _stable_clip(ypred) - ytrue

import numpy as np
from scipy.signal import correlate, convolve
from time import time, sleep
from os import listdir, path, makedirs
from PIL import Image
from random import sample
import struct

def no_effect(x):
    return x

def no_effect_deriv(x):
    return 1

def relu(x):
    return np.maximum(x, 0)

def relu_deriv(x):
    return x>0

def MSE(Y, Y_pred):
    dif = Y-Y_pred
    sq = dif**2
    loss = np.mean(sq)
    return loss

def MSE_deriv(Y, Y_pred):
    m = Y.size
    dY_pred = -(2/m)*(Y-Y_pred)
    return dY_pred

def softmax(a_all):
    y_all = np.zeros_like(a_all)
    for i, a in enumerate(a_all):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a/sum_exp_a
        y_all[i] = y
    return y_all

def softmax_deriv(a):
    softmaxed = softmax(a)
    return (1-softmaxed)*softmaxed

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))

def binary_cross_entropy_deriv(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    cost = ((1 - y_true) / (1 - y_pred + 1e-7) - y_true / (y_pred + 1e-7)) / np.size(y_true)
    return cost * 1000

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def load_images(dataset_dir, size, mode='RGB', limit=None):
    categories = listdir(dataset_dir)
    images = []
    labels = []
    for i, category in enumerate(categories):
        label = np.zeros((len(categories)))
        label[i] = 1
        titles = listdir(dataset_dir+'\\'+category)
        count = 0
        for title in titles:
            image = Image.open(dataset_dir+'\\'+category+'\\'+title).convert(mode)
            image = image.resize(size)
            image = np.array(image)
            if mode == 'RGB':
                image = np.reshape(image, (int(np.prod(np.array(size))), image.shape[-1]))
                image = image.T
                image = np.reshape(image, (image.shape[0], *size))
            elif mode == 'L':
                image = np.reshape(image, (1, *size))
            images.append(image)
            labels.append(label)
            count += 1
            if limit != None:
                if count >= limit:
                    break
    images = np.asarray(images)
    labels = np.asarray(labels)
    print('='*10+'image loading completed'+'='*10)
    return images, labels

class dense:
    def __init__(self, nods, activation=None, input_shape=None, trainable=True):
        self.type = 'dense'

        self.nods = nods
        self.input_shape = input_shape
        self.output_shape = 0
        self.activation_fn = activation

        if activation == None:
            self.activation = no_effect
            self.activation_deriv = no_effect_deriv
        elif activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_deriv = softmax_deriv

        self.x = np.array([])

        self.w = np.array([])
        self.b = np.array([])

        self.learning_rate = 0

        self.dw = np.array([])
        self.db = np.array([])

        self.rounder = False
        self.training_mode = True

        self.trainable = trainable

    def init2(self, input_shape, learning_rate=0.01, dtype=float, rounder=False, training_mode=True):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass

        w_shape_0 = self.input_shape[-1]
        self.w_shape = (w_shape_0, self.nods)
        self.b_shape = (self.nods, )
        self.output_shape = (self.nods, )

        self.w = np.random.randn(*self.w_shape).astype(dtype)
        self.b = np.zeros(self.b_shape, dtype=dtype)

        self.learning_rate = learning_rate

        self.rounder = rounder
        self.training_mode = training_mode

    def forward(self, x):
        self.x = x
        Z = np.add(x.dot(self.w), self.b)
        A = self.activation(Z)
        if self.rounder:
            A = np.round(A, 6)
        return Z, A

    def backward(self, dZ_next, previous_activation_derivative=1):
        m = len(self.x)
        self.dw = (1/m)*self.x.T.dot(dZ_next)
        self.db = (1/m)*sum(dZ_next)
        dz = dZ_next.dot(self.w.T)*previous_activation_derivative
        if self.rounder:
            self.dw = np.round(self.dw, 6)
            self.db = np.round(self.db, 6)
        return dz

    def apply_gradient(self):
        if self.trainable:
            self.w = self.w - self.learning_rate*self.dw
            self.b = self.b - self.learning_rate*self.db
            if self.rounder:
                self.w = np.round(self.w, 6)
                self.b = np.round(self.b, 6)

    def reverse_gradient(self):
        self.w = self.w + self.learning_rate*self.dw
        self.b = self.b + self.learning_rate*self.db

class conv2D:
    def __init__(self, filters, kernel_size, activation=None, input_shape=None, trainable=True):
        self.type = 'conv2D'

        self.filters = filters
        self.input_shape = input_shape
        self.output_shape = ()
        self.activation_fn = activation
        self.kernel_size = kernel_size

        if activation == None:
            self.activation = no_effect
            self.activation_deriv = no_effect_deriv
        elif activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_deriv = softmax_deriv

        self.x = np.array([])

        self.k = np.array([])
        self.b = np.array([])

        self.learning_rate = 0

        self.dk = np.array([])
        self.db = np.array([])

        self.rounder = False
        self.training_mode = True

        self.trainable = trainable

    def init2(self, input_shape, learning_rate=0.01, dtype=float, rounder=False, training_mode=True):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass

        k_shape = (self.filters, self.input_shape[0], *self.kernel_size)
        output_shape = (self.filters, int(self.input_shape[1]-self.kernel_size[0]+1), int(self.input_shape[1]-self.kernel_size[0]+1))
        self.output_shape = output_shape
        self.k = np.random.randn(*k_shape).astype(dtype)
        self.b = np.zeros(output_shape, dtype=dtype)

        self.learning_rate = learning_rate

        self.rounder = rounder
        self.training_mode = training_mode

    def forward(self, x):
        self.x = x
        Z = np.zeros((self.x.shape[0], *self.output_shape))
        for a, one_x in enumerate(self.x):
            for i in range(self.filters):
                for j in range(self.input_shape[0]):
                    Z[a][i] += correlate(self.x[a][j], self.k[i][j], mode='valid', method='direct')
                Z[a][i] += self.b[i]
        A = self.activation(Z)
        if self.rounder:
            A = np.round(A, 6)
        return Z, A

    def backward(self, dz_next, previous_activation_derivative):
        m = len(self.x)
        dk = np.zeros_like(self.k)
        dx = np.zeros_like(self.x)
        for a, one_x in enumerate(self.x):
            for i in range(self.filters):
                for j in range(self.input_shape[0]):
                    dk[i][j] += correlate(one_x[j], dz_next[a][i], mode='valid', method='direct')
                    dx[a][j] = convolve(dz_next[a][i], self.k[i][j], mode='full', method='direct')
        self.dk = dk/m
        self.db = sum(dz_next)/m
        dz = dx*previous_activation_derivative
        if self.rounder:
            self.dk = np.round(self.dk, 6)
            self.db = np.round(self.db, 6)
        return dz

    def apply_gradient(self):
        if self.trainable:
            self.k = self.k - self.learning_rate*self.dk
            self.b = self.b - self.learning_rate*self.db
            if self.rounder:
                self.k = np.round(self.k, 6)
                self.b = np.round(self.b, 6)

    def reverse_gradient(self):
        self.k = self.k + self.learning_rate*self.dk
        self.b = self.b + self.learning_rate*self.db

class avg_pool2D:
    def __init__(self, size, input_shape=None):
        self.type = 'avg_pool2D'

        self.input_shape = input_shape
        self.output_shape = ()
        self.activation_fn = None
        self.activation = no_effect
        self.activation_deriv = no_effect_deriv
        self.size = size

        self.k = np.ones(size)

        self.x = np.array([])

        self.learning_rate = 0

        self.rounder = False
        self.training_mode = True

    def init2(self, input_shape, learning_rate=0.01, dtype=float, rounder=False, training_mode=True):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass
        self.learning_rate = learning_rate
        self.output_shape = (self.input_shape[0], int(self.input_shape[1]-self.size[0]+1), int(self.input_shape[1]-self.size[0]+1))

        self.rounder = rounder
        self.training_mode = training_mode

    def forward(self, x):
        self.x = x
        Z = np.zeros((len(self.x), *self.output_shape))
        for a, one_x in enumerate(self.x):
            for i in range(one_x.shape[0]):
                Z[a][i] = correlate(one_x[i], self.k, mode='valid', method='direct')
        if self.rounder:
            Z = np.round(Z, 6)
        return Z, Z

    def backward(self, dz_next, previous_activation_derivative):
        m = self.size[0]
        dx = np.zeros_like(self.x)
        for a, one_dz_next in enumerate(dz_next):
            for i in range(self.input_shape[0]):
                dx[a][i] = correlate(one_dz_next[i], self.k, mode='full', method='direct')
        dx = dx/(m**2)
        dz = dx*previous_activation_derivative
        if self.rounder:
            dz = np.round(dz, 6)
        return dz

    def apply_gradient(self):
        pass

    def reverse_gradient(self):
        pass

class flatten:
    def __init__(self, input_shape=None):
        self.type = 'flatten'

        self.input_shape = input_shape
        self.output_shape = ()
        self.activation_fn = None
        self.activation = no_effect
        self.activation_deriv = no_effect_deriv

        self.x = np.array([])

        self.learning_rate = 0

        self.training_mode = True

    def init2(self, input_shape, learning_rate=0.01, dtype=float, rounder=False, training_mode=True):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass
        self.learning_rate = learning_rate
        self.output_shape = (int(np.prod(np.array(self.input_shape))), )
        self.training_mode = training_mode

    def forward(self, x):
        self.x = x
        Z = np.zeros((len(self.x), *self.output_shape))
        for a, one_x in enumerate(self.x):
            Z[a] = np.reshape(one_x, self.output_shape)
        return Z, Z

    def backward(self, dz_next, previous_activation_derivative):
        dz = np.zeros_like(self.x)
        for a, one_dz_next in enumerate(dz_next):
            dz[a] = np.reshape(one_dz_next, self.input_shape)
        return dz*previous_activation_derivative

    def apply_gradient(self):
        pass

    def reverse_gradient(self):
        pass

class dropout:
    def __init__(self, ratio, input_shape=None):
        self.type = 'dropout'

        self.ratio = ratio
        self.input_shape = input_shape
        self.output_shape = ()
        self.activation_fn = None
        self.activation = no_effect
        self.activation_deriv = no_effect_deriv

        self.x = np.array([])

        self.learning_rate = 0
        self.rounder = False

        self.training_mode = True
        self.arr = np.array([])

    def init2(self, input_shape, learning_rate=0.01, dtype=float, rounder=False, training_mode=True):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass
        self.learning_rate = learning_rate
        self.rounder = rounder
        self.training_mode = training_mode
        self.output_shape = input_shape

    def forward(self, x):
        self.x = x
        if self.training_mode:
            arr_size = self.x.size
            how_many_zeros = round(arr_size*self.ratio)
            self.arr = np.ones((arr_size))
            linear = np.linspace(0, arr_size - 1, arr_size, endpoint=True).tolist()
            seed = sample(linear, how_many_zeros)
            for i in seed:
                self.arr[i] = 0
            self.arr = np.reshape(self.arr, self.x.shape)
            Z = self.x*self.arr
        else:
            Z = self.x*(1-self.ratio)
        if self.rounder:
            Z = np.round(Z, 6)
        return Z, Z

    def backward(self, dz_next, previous_activation_derivative):
        dz = dz_next*self.arr
        if self.rounder:
            return np.round(dz, 6)
        else:
            return dz

    def apply_gradient(self):
        pass

    def reverse_gradient(self):
        pass

class model:
    def __init__(self, network, learning_rate=0.01, optimizer='gd', loss='mse', static_lr=True, mid_epoch_codes=None, dtype='float32', arduino=False, training_mode=True):
        self.network = network

        self.optimizer = optimizer

        self.input_shapes = []
        self.types = ['input']
        self.activations_deriv = [no_effect_deriv]
        for layer in self.network:
            self.input_shapes.append(layer.input_shape)
            self.types.append(layer.type)
            self.activations_deriv.append(layer.activation_deriv)
        for i in range(len(self.network)):
            try:
                self.network[i].init2(self.input_shapes[i], learning_rate, dtype=dtype, rounder=arduino, training_mode=training_mode)
                self.input_shapes[i+1] = self.network[i].output_shape
            except:
                pass

        self.loss_fn = None
        self.loss_fn_deriv = None
        if loss == 'mse':
            self.loss_fn = MSE
            self.loss_fn_deriv = MSE_deriv
        elif loss == 'binary_crossentropy':
            self.loss_fn = binary_cross_entropy
            self.loss_fn_deriv = binary_cross_entropy_deriv

        self.learning_rates = [0.1, 0.01, 0.001]
        self.static_lr = static_lr

        self.mid_epoch_codes_dir = mid_epoch_codes
        self.pauser = False
        self.repeat = True

    def forward_propagation(self, x):
        Zs, As = [x], [x]
        for layer in self.network:
            Z, A = layer.forward(As[-1])
            Zs.append(Z)
            As.append(A)
        return Zs, As

    def backward_propagation(self, x, y):
        Zs, As = self.forward_propagation(x)
        loss_deriv = self.loss_fn_deriv(y, As[-1])
        loss = self.loss_fn(y, As[-1])
        dZ = loss_deriv
        for i in range(len(self.network)):
            current_index = int(-(i+1))
            previous_index = int(-(i+2))
            previous_activation_derivative = self.activations_deriv[previous_index](Zs[previous_index])
            dZ = self.network[current_index].backward(dZ, previous_activation_derivative)
        return loss

    def predict(self, x):
        for i in range(len(self.network)):
            self.network[i].training_mode = False
        Zs, As = self.forward_propagation(x)
        prediction = As[-1]
        for i in range(len(self.network)):
            self.network[i].training_mode = True
        return prediction

    def accuracy_check(self, x, y):
        prediction = self.predict(x)
        correct = 0
        for i in range(len(prediction)):
            if np.argmax(prediction[i]) == np.argmax(y[i]):
                correct += 1
        accuracy = round((correct/len(x))*100, 3)
        print('accuracy : {} %'.format(accuracy))
        return accuracy

    def sgd(self, x, y, batch_size=32):
        size = x.shape[0]
        linear = np.linspace(0, size-1, size, endpoint=True).tolist()
        seed = sample(linear, batch_size)
        training_x = np.zeros((batch_size, *x.shape[1:]))
        training_y = np.zeros((batch_size, *y.shape[1:]))
        for i, a in enumerate(seed):
            training_x[i] = x[a]
            training_y[i] = y[a]
        cost = self.backward_propagation(training_x, training_y)
        return cost

    def update_learning_rate(self, lr):
        n = len(self.network)
        for i in range(n):
            self.network[i].learning_rate = lr

    def apply_grad(self):
        n = len(self.network)
        for i in range(n):
            self.network[i].apply_gradient()

    def reverse_grad(self):
        n = len(self.network)
        for i in range(n):
            self.network[i].reverse_gradient()

    def train(self, x, y, epochs=1, batch_size=32):
        for epoch in range(epochs):
            if self.mid_epoch_codes_dir != None:
                code = open(self.mid_epoch_codes_dir, 'r').read()
                exec(code)
            start = time()
            loss = 0
            if not self.pauser:
                if self.optimizer == 'gd':
                    loss = self.backward_propagation(x, y)
                elif self.optimizer == 'sgd':
                    loss = self.sgd(x, y, batch_size)
                if self.static_lr:
                    self.apply_grad()
                else:
                    costs = []
                    for lr in self.learning_rates:
                        self.update_learning_rate(lr)
                        self.apply_grad()
                        size = x.shape[0]
                        linear = np.linspace(0, size - 1, size, endpoint=True).tolist()
                        seed = sample(linear, batch_size)
                        testing_x = np.zeros((batch_size, *x.shape[1:]))
                        testing_y = np.zeros((batch_size, *y.shape[1:]))
                        for i, a in enumerate(seed):
                            testing_x[i] = x[int(a)]
                            testing_y[i] = y[int(a)]
                        pred = self.predict(x=testing_x)
                        loss2 = self.loss_fn(testing_y, pred)
                        costs.append(loss2)
                        self.reverse_grad()
                    minarg = np.argmin(np.array(costs))
                    self.update_learning_rate(self.learning_rates[int(minarg)])
                    self.apply_grad()
                    loss = costs[int(minarg)]
                end = time()
                duration = round(end-start, 3)
                print('epoch : {} | duration : {} seconds | loss : {}'.format(epoch+1, duration, loss))
            else:
                print('program is on pause')
                while self.repeat:
                    if self.mid_epoch_codes_dir != None:
                        code = open(self.mid_epoch_codes_dir, 'r').read()
                        exec(code)
                        sleep(1)
                print('program is now running')

    def pause(self):
        self.pauser = True
        self.repeat = True

    def release(self):
        self.pauser = False
        self.repeat = False

    def save(self, save_dir, dtype='float32'):
        if path.exists(save_dir) == False:
            makedirs(save_dir)

        shapes = open(save_dir+'\\shapes.txt', 'wb')
        shapes_txt = ''
        for count, layer in enumerate(self.network):
            if layer.type == 'dense':
                file1 = open(save_dir + '\\{}w'.format(count), 'wb')
                file1.write(layer.w.astype(dtype).tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.b.astype(dtype).tobytes())
                file2.close()

                shapes_txt += '{}\n'.format(layer.w.shape)
                shapes_txt += '{}\n'.format(layer.b.shape)
            elif layer.type == 'conv2D':
                file1 = open(save_dir + '\\{}k'.format(count), 'wb')
                file1.write(layer.k.astype(dtype).tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.b.astype(dtype).tobytes())
                file2.close()

                shapes_txt += '{}\n'.format(layer.k.shape)
                shapes_txt += '{}\n'.format(layer.b.shape)
            else:
                file1 = open(save_dir+'\\{}a'.format(count), 'wb')
                file1.close()
                file2 = open(save_dir+'\\{}b'.format(count), 'wb')
                file2.close()
                shapes_txt += 'a\n'
                shapes_txt += 'b\n'
        shapes.write(shapes_txt.encode())
        shapes.close()

    def load(self, save_dir, dtype='float32'):
        try:
            shapes = open(save_dir+'\\shapes.txt', 'rb').readlines()
            files = listdir(save_dir)
            files.remove('shapes.txt')
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for count, layer in enumerate(self.network):
                if layer.type == 'conv2D':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    kernel = np.frombuffer(file1, dtype=dtype)
                    kernel = np.reshape(kernel, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2, dtype=dtype)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.k = kernel
                    layer.b = bias
                elif layer.type == 'dense':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    weight = np.frombuffer(file1, dtype=dtype)
                    weight = np.reshape(weight, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2, dtype=dtype)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.w = weight
                    layer.b = bias
        except:
            pass

    def save_for_arduino(self, save_dir):
        if path.exists(save_dir) == False:
            makedirs(save_dir)

        shapes = open(save_dir + '\\shapes.txt', 'wb')
        shapes_txt = ''
        for count, layer in enumerate(self.network):
            if layer.type == 'dense':
                file1 = open(save_dir + '\\{}w'.format(count), 'wb')
                arr = np.round(layer.w, 6)
                arr = np.reshape(arr, (arr.size, ))
                byte_like = b''
                for i in arr:
                    byte_like += struct.pack('f', i)
                file1.write(byte_like)
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                arr = np.round(layer.b, 6)
                arr = np.reshape(arr, (arr.size,))
                byte_like = b''
                for i in arr:
                    byte_like += struct.pack('f', i)
                file2.write(byte_like)
                file2.close()

                shapes_txt += '{}\n'.format(layer.w.shape)
                shapes_txt += '{}\n'.format(layer.b.shape)
            elif layer.type == 'conv2D':
                file1 = open(save_dir + '\\{}k'.format(count), 'wb')
                arr = np.round(layer.k, 6)
                arr = np.reshape(arr, (arr.size,))
                byte_like = b''
                for i in arr:
                    byte_like += struct.pack('f', i)
                file1.write(byte_like)
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                arr = np.round(layer.b, 6)
                arr = np.reshape(arr, (arr.size,))
                byte_like = b''
                for i in arr:
                    byte_like += struct.pack('f', i)
                file2.write(byte_like)
                file2.close()

                shapes_txt += '{}\n'.format(layer.k.shape)
                shapes_txt += '{}\n'.format(layer.b.shape)
            else:
                file1 = open(save_dir+'\\{}a'.format(count), 'wb')
                file1.close()
                file2 = open(save_dir+'\\{}b'.format(count), 'wb')
                file2.close()
                shapes_txt += 'a\n'
                shapes_txt += 'b\n'
        shapes.write(shapes_txt.encode())
        shapes.close()

    def load_for_arduino(self, save_dir):
        try:
            shapes = open(save_dir+'\\shapes.txt', 'rb').readlines()
            files = listdir(save_dir)
            files.remove('shapes.txt')
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for count, layer in enumerate(self.network):
                if layer.type == 'conv2D':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb')
                    kernel = np.zeros(eval(shapes[int(2*count)]))
                    kernel = np.reshape(kernel, (kernel.size, ))
                    for i in range(kernel.size):
                        kernel[i] = struct.unpack('f', file1.read(4))[0]
                    kernel = np.reshape(kernel, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb')
                    bias = np.zeros(eval(shapes[int(2*count+1)]))
                    bias = np.reshape(bias, (bias.size, ))
                    for i in range(bias.size):
                        bias[i] = struct.unpack('f', file2.read(4))[0]
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.k = kernel
                    layer.b = bias
                elif layer.type == 'dense':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb')
                    weight = np.zeros(eval(shapes[int(2*count)]))
                    weight = np.reshape(weight, (weight.size, ))
                    for i in range(weight.size):
                        weight[i] = struct.unpack('f', file1.read(4))[0]
                    weight = np.reshape(weight, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb')
                    bias = np.zeros(eval(shapes[int(2*count+1)]))
                    bias = np.reshape(bias, (bias.size, ))
                    for i in range(bias.size):
                        bias[i] = struct.unpack('f', file2.read(4))[0]
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.w = weight
                    layer.b = bias
        except:
            pass
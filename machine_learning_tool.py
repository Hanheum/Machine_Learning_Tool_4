import cupy as np
from cupyx.scipy.signal import correlate, convolve
from time import time
from os import listdir
from PIL import Image
import math

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

def glorot_uniform(shape):
    np.random.seed(0)
    scale = 1/max(1., sum(shape)/2.)
    limit = math.sqrt(3.0*scale)
    weights = np.random.uniform(-limit, limit, size=shape)
    return weights

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
    def __init__(self, nods, activation=None, input_shape=None):
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

        self.w = 0
        self.b = 0

        self.learning_rate = 0

    def init2(self, input_shape, learning_rate=0.01):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass

        w_shape_0 = self.input_shape[-1]
        self.w_shape = (w_shape_0, self.nods)
        self.b_shape = (self.nods, )
        self.output_shape = (self.nods, )

        self.w = glorot_uniform(self.w_shape)
        self.b = np.zeros(self.b_shape)

        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        Z = np.add(x.dot(self.w), self.b)
        A = self.activation(Z)
        return Z, A

    def backward(self, dZ_next, previous_activation_derivative=1):
        m = len(self.x)
        dw = (1/m)*self.x.T.dot(dZ_next)
        db = (1/m)*sum(dZ_next)
        dz = dZ_next.dot(self.w.T)*previous_activation_derivative
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        return dz

class conv2D:
    def __init__(self, filters, kernel_size, activation=None, input_shape=None):
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

    def init2(self, input_shape, learning_rate=0.01):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass

        k_shape = (self.filters, self.input_shape[0], *self.kernel_size)
        output_shape = (self.filters, int(self.input_shape[1]-self.kernel_size[0]+1), int(self.input_shape[1]-self.kernel_size[0]+1))
        self.output_shape = output_shape
        self.k = glorot_uniform(k_shape)
        self.b = np.zeros(output_shape)

        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        Z = np.zeros((self.x.shape[0], *self.output_shape))
        for a, one_x in enumerate(self.x):
            for i in range(self.filters):
                for j in range(self.input_shape[0]):
                    Z[a][i] += correlate(self.x[a][j], self.k[i][j], mode='valid', method='direct')
                Z[a][i] += self.b[i]
        A = self.activation(Z)
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
        dk = dk/m
        db = sum(dz_next)/m
        dz = dx*previous_activation_derivative
        self.k = self.k - self.learning_rate*dk
        self.b = self.b - self.learning_rate*db
        return dz

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

    def init2(self, input_shape, learning_rate=0.01):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass
        self.learning_rate = learning_rate
        self.output_shape = (self.input_shape[0], int(self.input_shape[1]-self.size[0]+1), int(self.input_shape[1]-self.size[0]+1))

    def forward(self, x):
        self.x = x
        Z = np.zeros((len(self.x), *self.output_shape))
        for a, one_x in enumerate(self.x):
            for i in range(one_x.shape[0]):
                Z[a][i] = correlate(one_x[i], self.k, mode='valid', method='direct')
        return Z, Z

    def backward(self, dz_next, previous_activation_derivative):
        m = self.size[0]
        dx = np.zeros_like(self.x)
        for a, one_dz_next in enumerate(dz_next):
            for i in range(self.input_shape[0]):
                dx[a][i] = correlate(one_dz_next[i], self.k, mode='full', method='direct')
        dx = dx/(m**2)
        dz = dx*previous_activation_derivative
        return dz

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

    def init2(self, input_shape, learning_rate=0.01):
        try:
            if self.input_shape == None:
                self.input_shape = input_shape
        except:
            pass
        self.learning_rate = learning_rate
        self.output_shape = (int(np.prod(np.array(self.input_shape))), )

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

class model:
    def __init__(self, network, learning_rate=0.01, optimizer='gd', loss='mse'):
        self.network = network

        self.input_shapes = []
        self.types = ['input']
        self.activations_deriv = [no_effect_deriv]
        for layer in self.network:
            self.input_shapes.append(layer.input_shape)
            self.types.append(layer.type)
            self.activations_deriv.append(layer.activation_deriv)
        for i in range(len(self.network)):
            try:
                self.network[i].init2(self.input_shapes[i], learning_rate)
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
        Zs, As = self.forward_propagation(x)
        prediction = As[-1]
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

    def train(self, x, y, epochs=1):
        for epoch in range(epochs):
            start = time()
            loss = self.backward_propagation(x, y)
            end = time()
            duration = round(end-start, 3)
            print('epoch : {} | duration : {} seconds | loss : {}'.format(epoch+1, duration, loss))

    def save(self, save_dir):
        shapes = open(save_dir+'\\shapes.txt', 'wb')
        shapes_txt = ''
        for count, layer in enumerate(self.network):
            if layer.type == 'dense':
                file1 = open(save_dir + '\\{}w'.format(count), 'wb')
                file1.write(layer.w.tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.b.tobytes())
                file2.close()

                shapes_txt += '{}\n'.format(layer.w.shape)
                shapes_txt += '{}\n'.format(layer.b.shape)
            elif layer.type == 'conv2D':
                file1 = open(save_dir + '\\{}k'.format(count), 'wb')
                file1.write(layer.k.tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.b.tobytes())
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

    def load(self, save_dir):
        try:
            shapes = open(save_dir+'\\shapes.txt', 'rb').readlines()
            files = listdir(save_dir)
            files.remove('shapes.txt')
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for count, layer in enumerate(self.network):
                if layer.type == 'conv2D':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    kernel = np.frombuffer(file1)
                    kernel = np.reshape(kernel, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.k = kernel
                    layer.b = bias
                elif layer.type == 'dense':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    weight = np.frombuffer(file1)
                    weight = np.reshape(weight, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.w = weight
                    layer.b = bias
        except:
            pass
import machine_learning_tool as ml
import cupy as np
from PIL import Image

latent_dim = 64

network = [
    ml.flatten(input_shape=(1, 28, 28)),
    ml.dense(latent_dim, activation='relu'),
    ml.dense(784, activation='sigmoid')
]

model = ml.model(network=network)
model.load('C:\\Users\\chh36\\Desktop\\ae_test')

while True:
    image = Image.open('C:\\Users\\chh36\\Desktop\\mini_mnist_things\\mini_mnist\\{}'.format(input('img:'))).convert('L')
    image = np.array(image)
    image = np.reshape(image, (1, 1, 28, 28))/255.0
    prediction = model.predict(image)*255.0
    prediction = np.reshape(prediction, (28, 28))
    prediction = prediction.astype('uint8')
    prediction = np.asnumpy(prediction)
    prediction = Image.fromarray(prediction)
    prediction.show()
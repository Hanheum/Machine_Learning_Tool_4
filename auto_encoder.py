import machine_learning_tool as ml
import cupy as np

images, _ = ml.load_images('C:\\Users\\chh36\\Desktop\\mini_mnist_things\\mini_mnist', size=(28, 28), mode='L')
images = images.astype('float32')/255.0

latent_dim = 64

network = [
    ml.flatten(input_shape=(1, 28, 28)),
    ml.dense(latent_dim, activation='relu'),
    ml.dense(784, activation='sigmoid')
]

model = ml.model(network=network, loss='mse', optimizer='gd', learning_rate=10000)
model.load('C:\\Users\\chh36\\Desktop\\ae_test')
model.train(x=images, y=np.reshape(images, (images.shape[0], 784)), epochs=300)

model.save('C:\\Users\\chh36\\Desktop\\ae_test')
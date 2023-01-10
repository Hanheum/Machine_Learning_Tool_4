import machine_learning_tool as ml

imgs, labels = ml.load_images('C:\\Users\\chh36\\Desktop\\turret\\primary', (32, 32), limit=None, mode='L')

network = [
    ml.conv2D(filters=10, kernel_size=(5, 5), activation='relu', input_shape=(1, 32, 32)),
    ml.avg_pool2D(size=(3, 3)),
    ml.flatten(),
    ml.dense(2, activation='softmax')
]

model = ml.model(network=network, learning_rate=0.001, loss='mse', optimizer='sgd')
model.load('C:\\Users\\chh36\\Desktop\\primary_model2')

model.accuracy_check(imgs, labels)
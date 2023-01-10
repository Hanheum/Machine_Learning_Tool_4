import machine_learning_tool as ml

images, labels = ml.load_images('C:\\Users\\chh36\\Desktop\\turret\\primary', size=(32, 32), mode='L', limit=100)
images = images/255.0

network = [
    ml.conv2D(filters=10, kernel_size=(3, 3), activation='relu', input_shape=(1, 32, 32)),
    ml.conv2D(filters=10, kernel_size=(3, 3), activation='relu'),
    ml.flatten(),
    ml.dense(nods=2, activation='softmax')
]

model = ml.model(network=network, loss='mse', optimizer='gd', learning_rate=0.01, static_lr=True, mid_epoch_codes='C:\\Users\\chh36\\Desktop\\mid_epoch_code.txt')
model.load('C:\\Users\\chh36\\Desktop\\deep2')
model.accuracy_check(images, labels)
model.train(x=images, y=labels, epochs=100, batch_size=200)
model.save('C:\\Users\\chh36\\Desktop\\deep2')
model.accuracy_check(images, labels)
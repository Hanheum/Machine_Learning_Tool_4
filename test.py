import machine_learning_tool as ml

dataset_dir = 'C:\\Users\\chh36\\Desktop\\mini_mnist_things\\mini_mnist\\'
images, labels = ml.load_images(dataset_dir, (28, 28), mode='L')

network = [
    ml.conv2D(filters=5, kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28)),
    ml.avg_pool2D(size=(2, 2)),
    ml.conv2D(filters=5, kernel_size=(3, 3), activation='relu'),
    ml.avg_pool2D(size=(2, 2)),
    ml.flatten(),
    ml.dense(nods=10, activation='relu'),
    ml.dense(nods=2, activation='softmax')
]

model = ml.model(network, learning_rate=0.001, loss='binary_crossentropy')
#model.load('C:\\Users\\chh36\\Desktop\\ml_tool\\zero_one_test-99% accuracy')

model.accuracy_check(images, labels)
model.train(x=images, y=labels, epochs=10)
model.accuracy_check(images, labels)

#model.save('C:\\Users\\chh36\\Desktop\\zero_one_test')
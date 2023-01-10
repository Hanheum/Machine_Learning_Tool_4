import machine_learning_tool as ml
import cupy as np

x = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

x = np.asarray(x)
y = np.asarray(y)

network = [
    ml.dense(nods=100, activation='relu', input_shape=(2, )),
    ml.dense(nods=100, activation='relu'),
    ml.dense(nods=3, activation='softmax')
]

model = ml.model(network=network, loss='mse', learning_rate=0.01)
model.accuracy_check(x, y)
model.train(x=x, y=y, epochs=10000)
model.accuracy_check(x, y)
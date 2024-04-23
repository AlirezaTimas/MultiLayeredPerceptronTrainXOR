from keras.models import Sequential
from keras.layers import Dense #estefade az layer type dense bekhater fullyconnected boodan
import numpy as np

# Voroodi ha jahat train (XOR table)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Tolid yek sequential model
model = Sequential()

# Ezafe kardan hidden layer be 2 neuron va activation function rectified linear unit
model.add(Dense(2, input_dim=2, activation='relu'))

# Ezafe kardan laye khorooji ba 1 neuron va activation function sigmoid (:estefade az sigmoid jahat inke khorooji beine 0 va 1 bashad)
model.add(Dense(1, activation='sigmoid'))

# compile kardan model ba adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model baraye 1000 epoch
model.fit(X_train, y_train, epochs=1000, verbose=0)

# testesh
X_test = np.array([[0, 1], [0, 0] , [1,1] , [1,0]])
predictions = model.predict(X_test)

# inam prediction model :round vase inke be 0 o 1 round she chon XOR binary value javab mide
print("Predictions:", predictions.round())
#AlirezaTimas
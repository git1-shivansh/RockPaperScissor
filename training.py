import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Conv2D, GlobalAveragePooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import os

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model(input_shape, num_classes):
    model = Sequential([
        SqueezeNet(input_shape=input_shape, include_top=False),
        Dropout(0.5),
        Conv2D(num_classes, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


def load_data(img_dir):
    dataset = []
    for directory in os.listdir(img_dir):
        path = os.path.join(img_dir, directory)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            # to make sure no hidden files get in our way
            if item.startswith("."):
                continue
            img = cv2.imread(os.path.join(path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (227, 227))
            dataset.append([img, directory])

    return dataset


def preprocess_data(dataset, class_map):
    data, labels = zip(*dataset)
    labels = list(map(mapper, labels))
    # one hot encode the labels
    labels = np_utils.to_categorical(labels, num_classes=len(class_map))

    return np.array(data), np.array(labels)


def train_model(model, X_train, Y_train, epochs=10, batch_size=32):
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # start training
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    return model


def main():
    # load the data
    dataset = load_data(IMG_SAVE_PATH)

    # preprocess the data
    X, Y = preprocess_data(dataset, CLASS_MAP)

    # split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # create the model
    input_shape = X_train.shape[1:]
    num_classes = Y_train.shape[1]
    model = get_model(input_shape, num_classes)

    # train the model
    model = train_model(model, X_train, Y_train, epochs=10, batch_size=32)

    # evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # save the model for later use
    model.save("rock-paper-scissors-model.h5")


if __name__ == '__main__':
    main()

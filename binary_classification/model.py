from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential


def create_model(image_dimensions):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(image_dimensions[0], image_dimensions[1], 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

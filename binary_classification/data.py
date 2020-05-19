from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_training_data_generators(train_dir, validation_dir, batch_size=128, image_dimensions=(150, 150)):
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=image_dimensions, class_mode='binary')

    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    validation_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                         directory=validation_dir,
                                                                         target_size=image_dimensions,
                                                                         class_mode='binary')

    return train_data_gen, validation_data_gen

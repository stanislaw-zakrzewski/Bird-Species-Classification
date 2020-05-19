import argparse
import os

from tensorflow.keras.models import load_model

from binary_classification.data import create_training_data_generators
from binary_classification.model import create_model
from utils.charts import visualize_training

# Default values
PATH = '2-bird-species/'
TRAIN_DIR = os.path.join(PATH, 'train')
VALIDATION_DIR = os.path.join(PATH, 'valid')
TEST_DIR = os.path.join(PATH, 'test')

BATCH_SIZE = 128
EPOCHS = 100
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument('--load_from', type=str, help='Use model trained previously')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='How many epochs in training process')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT, help='Image height')
    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH, help='Image width')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR, help='Training images directory')
    parser.add_argument('--validation_dir', type=str, default=VALIDATION_DIR, help='Validation images directory')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help='Test images directory')
    return parser.parse_args()


# Read run arguments
config = get_arguments()
image_dimensions = (config.image_height, config.image_width)

# Create train, validation and test test data generators
train_data_gen, val_data_gen, test_data_gen = create_training_data_generators(config.train_dir, config.validation_dir,
                                                                              config.test_dir,
                                                                              config.batch_size,
                                                                              image_dimensions=image_dimensions)

if config.load_from is not None:
    model = load_model(config.load_from)
else:
    # Create model with dropouts
    model = create_model(image_dimensions)

    # Train model
    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_data_gen.samples // config.batch_size,
        epochs=config.epochs,
        validation_data=val_data_gen,
        validation_steps=val_data_gen.samples // config.batch_size
    )

    # Save trained model
    model.save('saved_model_binary_classifier')

    # Show training progress
    visualize_training(config.epochs, history)

test_loss, test_acc = model.evaluate_generator(test_data_gen, test_data_gen.samples)
print(test_loss, test_acc)

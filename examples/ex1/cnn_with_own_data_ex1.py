import os
import tarfile

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

# We can verify that we have a GPU before we start running:
print(tf.config.list_physical_devices('GPU'))


# Unpack archive on $TMPDIR to reduce common file IO load
# Note, that this could just as well have happened in the jobscript
tmpdir = os.getenv("TMPDIR")
with tarfile.open("data.tar.gz", "r:gz") as data_archive:
    data_archive.extractall(tmpdir)
datadir = f"{tmpdir}/data"

# Load data
train_batches = ImageDataGenerator().flow_from_directory(
    datadir,
    target_size=(10, 10),
    color_mode='grayscale',
    classes=[str(ix) for ix in range(1, 11)],
    batch_size=128,
)

#imgs, labels = next(train_batches)

# In defining the model, the input shape must match the dimension of the input
# data. We have grayscale images, therefore, the input_shape is (xx, xx, 1),
# and the ImageDataGenerator should also be aware of that too:
# color_mode='grayscale'

model = Sequential([
    layers.Conv2D(10, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'],
)

model.summary()

model.fit(train_batches, epochs=3, verbose=2)

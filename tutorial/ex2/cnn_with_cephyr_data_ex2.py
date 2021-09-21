from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print("Loading dataset...")
img_path = '/cephyr/NOBACKUP/Datasets/tiny-imagenet-200/train'

train_batches = ImageDataGenerator().flow_from_directory(
    img_path, target_size=(64, 64), color_mode="rgb", batch_size=128)
print("done.")

# In defining the model, the input shape must match the dimension of the input
# data. We have rgb-color images, therefore, the input_shape is (xx, xx, 3),
# and the ImageDataGenerator should also be aware of that too: color_mode='rgb'

model = Sequential([layers.Conv2D(10, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                    layers.Flatten(),
                    layers.Dense(200, activation='softmax'),
                    ])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_batches, steps_per_epoch=781, epochs=10, verbose=2)

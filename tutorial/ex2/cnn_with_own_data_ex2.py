import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

img_path = '/cephyr/NOBACKUP/Datasets/YouTube-dataset/YouTubeVos_InstanceSegmentation_2019/train_all_frames/JPEGImages'
train_batches = ImageDataGenerator().flow_from_directory(img_path, target_size=(10, 10), color_mode='grayscale',                                                         
                                                         batch_size=13)

#imgs, labels = next(train_batches)

# In defining the model, the input shape must match the dimension of the input data. We have grayscale images, 
# therefore, the input_shape is (xx, xx, 1), and the ImageDataGenerator should also 
# be aware of that too: color_mode='grayscale'

model = Sequential([layers.Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 1)), 
                                 layers.Flatten(),
                    layers.Dense(3471, activation='softmax'),
                        
    
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_batches, steps_per_epoch=267, epochs=10, verbose=2)

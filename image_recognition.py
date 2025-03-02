import os
import shutil
import requests
import pendulum
from pathlib import Path
import hashlib

################### start model imports

import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
import numpy as np
from IPython.display import Image

#################### end model imports


class ImageDetector:
    photo_data_url = 'https://nta.local/api/get-photo-dates'
    photo_folder = '/mnt/d/temp/ImageDetection/'
    mobile = keras.applications.mobilenet.MobileNet()

    def __init__(self):
        self.model = None

    def rename_photos(self):
        photo_sessions = requests.get(self.photo_data_url, verify=False).json()
        last_key = len(photo_sessions) - 1

        for i in range(last_key + 1):
            article_time = pendulum.parse(photo_sessions[i].get('Timestamp'))
            for file in os.listdir(self.photo_folder + 'new'):
                full_path = self.photo_folder + 'new/' + file
                path = Path(full_path)
                ts_parts = path.stem.split('_')
                if len(ts_parts) < 3:
                    continue
                ts = " ".join(ts_parts[1:])
                file_time = pendulum.parse(ts)
                is_after = file_time > article_time
                is_before = True
                if i < last_key:
                    is_before = file_time < pendulum.parse(photo_sessions[i + 1].get('Timestamp'))
                if is_after and is_before:
                    random = hashlib.md5(file.encode()).hexdigest()[0:3]

                    dest_folder = self.photo_folder + 'train/' + photo_sessions[i]['Artikel']
                    os.makedirs(dest_folder, exist_ok=True)

                    shutil.copy(full_path, dest_folder)
                    shutil.move(dest_folder + '/' + path.name, dest_folder + '/' + random + '.jpg')


    def prepare_image(self, file):
        img = image.load_img(self.photo_folder + '/' + file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    def rebuild_model(self):
        base_model = keras.applications.MobileNet(weights='imagenet', include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
        x = Dense(1024, activation='relu')(x)  # dense layer 2
        x = Dense(512, activation='relu')(x)  # dense layer 3
        preds = Dense(3, activation='softmax')(x)  # final layer with softmax activation

        self.model = Model(inputs=base_model.input, outputs=preds)
        for layer in self.model.layers[:-5]:
            layer.trainable = False

        train_datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet.preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(0, 0.2)
        )

        train_generator = train_datagen.flow_from_directory(
            directory=self.photo_folder + 'train',
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=True
        )

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Adam optimizer
        # loss function will be categorical cross entropy
        # evaluation metric will be accuracy

        step_size_train = train_generator.n // train_generator.batch_size
        self.model.fit(train_generator, steps_per_epoch=step_size_train, epochs=5)

        self.model.export('saved_model/article_model')
        self.save_class_names(self.photo_folder + 'train', 'saved_model/article_model')


    def save_class_names(self, train_dir, save_dir):
        classes = []

        for subdir in sorted(os.listdir(train_dir)):
            if os.path.isdir(os.path.join(train_dir, subdir)):
                classes.append(subdir)

        with open(os.path.join(save_dir, 'class_names.txt'), 'w') as f:
            f.write(','.join(classes))


    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.  # imshow expects values in the range [0, 1]

        return img_tensor


    def test(self):
        # self.rename_photos()
        # self.rebuild_model()
        self.save_class_names(self.photo_folder + 'train', 'saved_model/article_model')
        # self.model = keras.layers.TFSMLayer('saved_model/article_model', call_endpoint='serving_default')
        # self.model = keras.models.load_model('saved_model/article_model')


        # img_path = '/mnt/d/temp/ImageDetection/test/aaa_nose_spray.jpg'
        # img_path = '/mnt/d/temp/ImageDetection/test/bbb_tea_tree_oil.jpg'
        # img_path = '/mnt/d/temp/ImageDetection/test/ccc_measuring_cup.jpg'
        # new_image = self.load_image(img_path)

        # pred = self.model.predict(new_image)
        # print(pred)


image_detector = ImageDetector()
image_detector.test()
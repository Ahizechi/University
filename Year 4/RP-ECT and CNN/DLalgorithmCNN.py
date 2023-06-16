import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Concatenate
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from tqdm import tqdm


def gather_data(path):
    data = []
    labels = []
    shape_folders = os.listdir(path)
    total_folders = sum([len(os.listdir(os.path.join(path, shape))) for shape in shape_folders])

    with tqdm(total=total_folders, desc="Gathering data") as pbar:
        for shape in shape_folders:
            shape_path = os.path.join(path, shape)
            image_count = 0
            for individual_folder in os.listdir(shape_path):
                individual_path = os.path.join(shape_path, individual_folder)
                if os.path.isdir(individual_path):
                    images = []
                    for image_name in os.listdir(individual_path):
                        image_path = os.path.join(individual_path, image_name)
                        img = cv2.imread(image_path)
                        if img is not None:
                            img = cv2.resize(img, (100, 100))
                            images.append(img)
                        else:
                            print(f"Unable to read image: {image_path}")

                    if len(images) == 16:
                        data.append(images)
                        labels.append(shape)
                        image_count += 1
                    else:
                        print(f"Folder {individual_path} does not contain 16 images")

                pbar.update(1)
            
            # print(f"Total images gathered for {shape}: {image_count}")

    return np.array(data), np.array(labels)


def create_dl_model(num_classes):
    input_layers = []
    output_layers = []
    
    for _ in range(16):
        input_img = Input(shape=(100, 100, 3))
        x = Conv2D(32, (3, 3), activation='relu')(input_img)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)  # Add a dropout layer
        input_layers.append(input_img)
        output_layers.append(x)
    
    x = Concatenate()(output_layers)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)  # Add a dropout layer
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layers, outputs=x)
    return model

def custom_generator(X, y, batch_size, augment=True):
    num_samples = len(X[0])
    datagen = ImageDataGenerator(rotation_range=50, height_shift_range=0.3, horizontal_flip=True)
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = [X[i][start:end] / 255.0 for i in range(16)]
            y_batch = y[start:end]
            if augment:
                X_batch = [datagen.flow(x, batch_size=len(x), shuffle=False).next() for x in X_batch]
            yield X_batch, y_batch


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Compile the model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    batch_size = 32
    train_generator = custom_generator(X_train, y_train, batch_size, augment=True)
    test_generator = custom_generator(X_test, y_test, batch_size)
    model.fit(train_generator, epochs=50, validation_data=test_generator, steps_per_epoch=len(X_train[0]) // batch_size, validation_steps=len(X_test[0]) // batch_size)

    # Evaluate the model
    predictions = model.predict(X_test, batch_size=batch_size)
    report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1))
    return report, predictions

def save_results(predictions, X_test, lb, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for idx, pred in tqdm(enumerate(predictions.argmax(axis=1)), desc="Saving results", total=len(predictions)):
        shape_folder = lb.classes_[pred]
        shape_result_folder = os.path.join(result_folder, shape_folder)
        if not os.path.exists(shape_result_folder):
            os.makedirs(shape_result_folder)

        img_name = f"{shape_folder}_image_{idx}.png"
        img_path = os.path.join(shape_result_folder, img_name)
        cv2.imwrite(img_path, X_test[0][idx])


def main():
    # Step 1: Gather data
    data_folder = 'D:\\University\\Year 4\\Project\\Simulation\\Results_Final'
    X, y = gather_data(data_folder)
    # Encode the labels using LabelBinarizer
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Step 2: Create the deep learning model
    model = create_dl_model(len(lb.classes_))

    # Prepare data for multi-input model
    X_train = [X_train[:, i] for i in range(16)]
    X_test = [X_test[:, i] for i in range(16)]

    # Step 3: Train and evaluate the model
    report, predictions = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    print(report)

    # Step 4: Save the results
    result_folder = 'D:\\University\\Year 4\\Project\\Simulation\\DLResultsCNN'
    save_results(predictions, X_test, lb, result_folder)

    # Step 5: Save the model
    model.save("model.h5")

if __name__ == "__main__":
    main()


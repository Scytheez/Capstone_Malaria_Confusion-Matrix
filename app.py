import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model

img_height = 180
img_width = 180
batch_size = 32 

val_count = []
inval_count = []

def validate():
    try:
        dataset_path = '../../Documents/Malaria Dataset/validation dataset' # ChangeMe
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        class_names = train_ds.class_names

        val_model_path = 'trained_model/val_model.keras' # Changeme
        model = load_model(val_model_path)

        #model.evaluate(train_ds)
        model.summary() 

        #input_dir = 'parasitized_cell'
        input_dir = 'uninfected_cell' # Changme

        print('VALIDATE IMAGE')
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                confidence = ("{:.2f}".format(100 * np.max(score)))

                print("Image: ", filename)
                print(f'{class_names[np.argmax(score)]} = {score}')
                print(f"Confidence Level: {confidence}")
                print("------------------------------------")

                predicted_class = class_names[np.argmax(score)]

                if predicted_class == 'invalid':
                    inval_count.append(filename)
                elif predicted_class == 'valid':
                    val_count.append(filename)
                else:
                    print('Something went wrong')

    except Exception as e:
        print(e)

    finally:
        print()
        print(f'Valid: {len(val_count)}')
        print(f'Invalid: {len(inval_count)}')
        print()
        print('===============================================================')
        print()
        print()


def main():
    try:
        # Dataset path
        dataset_path = '../../Documents/Malaria Dataset/Dataset Pixel'
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        img_height = 180
        img_width = 180
        batch_size = 32

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names

        # User input image
        input_dir = "parasitized_cell"
        #input_dir = "uninfected_cell"
        # Saved model path
        model_path = "trained_model/my_model.keras"

        model = load_model(model_path)
        #model.evaluate(train_ds)
        model.summary()

        un_count = []
        par_count = []
        overall = []

        for filename in os.listdir(input_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                predicted_class = class_names[np.argmax(score)]

                print("Image: ", filename)
                if predicted_class == 'Uninfected':
                    print('Prediction: Uninfected Cell')
                    print("Confidence Level: {:.2f}%".format(100 * np.max(score)))
                    un_count.append(filename)
                    overall.append(predicted_class)
                    #np.append(overall, [predicted_class])
                    print("------------------------------------")
                elif predicted_class == 'Parasitized':
                    print('Prediction: Infected Cell')
                    print("Confidence Level: {:.2f}%".format(100 * np.max(score)))
                    par_count.append(filename)
                    overall.append(predicted_class)
                    #np.append(overall, [predicted_class])
                    print("------------------------------------")
                else:
                    print('Not a type of sample!')
    except Exception as e:
        print(e)

    print(f'Uninfected: {len(un_count)}')
    print(f'Parasitized: {len(par_count)}')
    #print(overall, len(overall))
    print('------------------------------------------------------')

    # CONFUSION MATRIX
    import seaborn as sns
    sns.set_style('darkgrid')
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=333, n_features=20, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #y_test = np.array([])
    y_test = []
    for i in range(0,100):
        #y_test.append(['Uninfected'])
        y_test.append(['Parasitized'])
    #print(xy_test, len(xy_test))

    cm = confusion_matrix(y_test, overall, labels=['Parasitized', 'Uninfected']) #y_test | y_pred
    print(f'Confusion Matrix Value: {cm} | {cm.shape}')

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Parasitized', 'Uninfected'], yticklabels=['Parasitized', 'Uninfected'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)

    print(classification_report(y_test, overall)) #y_test | y_pred | , target_names=class_names

    plt.show()

if __name__ == '__main__':
    validate()
    main()
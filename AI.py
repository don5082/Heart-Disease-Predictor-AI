import pandas as pd
from flatbuffers.packer import int64
from sklearn.model_selection import train_test_split
import tensorflow as tf

NUM_EPOCHS = 1000

def main():
    dataset = pd.read_csv('heart.csv')

    custom_mappings = {
        'Sex': {'M': 0, 'F': 1},
        'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
        'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 2, 'Flat': 1, 'Down': 0}
    }

    for col, mapping in custom_mappings.items():
        dataset[col] = dataset[col].map(mapping)

    x = dataset.drop(columns=["HeartDisease"])
    y = dataset["HeartDisease"]

    x = x.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=256, input_shape=(x_train.shape[1:]), activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=NUM_EPOCHS)

    model.evaluate(x_test, y_test)

    # print(x)
    # print(y)

    # print(x_train.dtypes)
    # print(y_train.dtypes)

if __name__ == '__main__':
    main()
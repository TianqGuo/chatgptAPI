from keras.backend import dropout
import pandas as pd
import numpy as np
import tensorflow as tf

def test():
    model = tf.keras.models.load_model('my_model.h5')
    config = model.get_config() # Returns pretty much every information about your model
    print(config)
    # print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    df = pd.read_csv('preprocessed_kevin')
    y = ['metformin', 'repaglinide','nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone','rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin','glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone', 'metformin-pioglitazone']
    x = df.drop(y, axis = 1)
    y = df[y]
    test_input = np.array(x.iloc[0])
    test_input = test_input[None, :] # shape 1x4
    print(test_input.shape)

    #print(test_input)
    test_output = model.predict(test_input)
    print(test_output)
    print(test_output.shape)

if __name__ == "__main__":
    test()


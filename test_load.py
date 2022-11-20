import unittest
import json

import tensorflow as tf

from model_functions import get_keras_model


class LoadTest(unittest.TestCase):
    def test_data_loading(self):
        data_sample = 'This is test row'
        dataset = {
            'samples':
                [
                    {
                        'text': data_sample
                    }
                ]
        }
        with open('test.json', 'w') as outfile:
            json.dump(dataset, outfile)
        with open('test.json', "r") as inpfile:
            ds = json.load(inpfile)

        self.assertEqual(dataset, ds)

    def test_model_loading(self):
        model1 = get_keras_model()
        tf.keras.models.save_model(model1, 'sample_model')
        model2 = tf.keras.models.load_model('sample_model')
        self.assertEqual(model1.get_config(), model2.get_config())


if __name__ == '__main__':
    unittest.main()

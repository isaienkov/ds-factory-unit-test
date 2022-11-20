import unittest

from data_functions import preprocess_pandas_df, preprocess_image, get_mnist_image


class DataTest(unittest.TestCase):
    def test_preprocess_pandas_df(self):
        expected = ['A', 'B', 'D']
        output = preprocess_pandas_df()
        self.assertEqual(expected, list(output.columns))

    def test_preprocess_image(self):
        img = get_mnist_image()
        img = preprocess_image(img)
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(img.max(), 1)


if __name__ == '__main__':
    unittest.main()

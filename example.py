import unittest


class ExampleTest(unittest.TestCase):
    def setUp(self):
        print('setUp')

    def tearDown(self):
        print('tearDown')

    @classmethod
    def setUpClass(cls):
        print('setUpClass')

    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def test_1(self):
        print('test1')
        self.assertEqual(1, 1)

    def test_2(self):
        print('test2')
        self.assertEqual(2, 2)

    def test_3(self):
        print('test3')
        self.assertEqual(3, 3)


if __name__ == '__main__':
    unittest.main()

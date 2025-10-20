import unittest
from my_math import add, factorial

class TestAddFunction(unittest.TestCase):
    
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1,-1),-2)
    def test_base_case(self):
        self.assertEqual(factorial(0),1)
    def test_standard_case(self):
        self.assertEqual(factorial(5),120)
    def test_negative_input(self):
        with self.assertRaises(ValueError):
            factorial(-1)
            
if __name__ == '__main__':
    unittest.main()
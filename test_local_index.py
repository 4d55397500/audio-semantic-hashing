import unittest

import local_index
import custom_exceptions


class TestLocalIndex(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_index(self):
        try:
            local_index.create_index()
        except custom_exceptions.ModelNotFoundException:
            pass


if __name__ == "__main__":
    unittest.main()

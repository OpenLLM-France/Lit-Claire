import os
import sys
import unittest

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from unittests import *

if __name__ == '__main__':
    unittest.main()

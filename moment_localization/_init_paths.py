# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Details: import other paths
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.getcwd()

lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)






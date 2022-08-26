import unittest

import numpy as np

from skesn.misc import correct_dimensions


class MiscFunctionalCheck(unittest.TestCase):
    y_matrices = [
        {
            'original': np.array([1., 2., 3., 4.]),
            '2D': np.array([
                [1.],
                [2.],
                [3.],
                [4.]
            ]),
            '3D': np.array([[
                [1.],
                [2.],
                [3.],
                [4.]
            ]]),
        },
        {
            'original': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '2D': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '3D': np.array([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]]),
        },
        {
            'original': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
            '2D': None,
            '3D': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
        },
    ]
    X_matrices = [
        {
            'original': None,
            '2D': None,
            '3D': None,
        },
        {
            'original': np.array([1., 2., 3., 4.]),
            '2D': np.array([
                [1.],
                [2.],
                [3.],
                [4.]
            ]),
            '3D': np.array([[
                [1.],
                [2.],
                [3.],
                [4.]
            ]]),
        },
        {
            'original': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '2D': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '3D': np.array([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]]),
        },
        {
            'original': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
            '2D': None,
            '3D': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
        },
    ]

    def test_correct_dimensions(self):
        for y in self.y_matrices + self.X_matrices:
            for repr in ('2D', '3D'):
                y_corrected = correct_dimensions(y['original'], representation=repr)
                if y[repr] is None:
                    self.assertEqual(y_corrected, y[repr])
                else:
                    np.testing.assert_allclose(y_corrected, y[repr])

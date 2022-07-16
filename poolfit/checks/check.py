
import torch as pt
import tensorflow as tf

import numpy as np
import quaternion

def test_match(p,q):
    print("=================")
    p = p.numpy()
    q = q.numpy()
    if np.allclose(p,q):
        print("OK")
    else:
        np.testing.assert_allclose(p,q)
    print("=================")

N=10
np_axis_angle = np.random.random( (N,3) ) - 0.5
np_axis_angle[5:]*=0.001

import pytorch_impl.rotation_conversion as rcpt
pt_axis_angle  = pt.from_numpy(np_axis_angle).to(pt.float32)
mat1A = rcpt.axis_angle_to_quaternion(pt_axis_angle)
mat1B = rcpt.axis_angle_to_matrix(pt_axis_angle)
mat1C = rcpt.matrix_to_axis_angle(mat1B)

import tensorlow_impl.rotation_conversion as rctf
tf_axis_angle  = tf.convert_to_tensor(np_axis_angle, dtype=tf.float32)
mat2A = rctf.axis_angle_to_quaternion(tf_axis_angle)
mat2B = rctf.axis_angle_to_matrix(tf_axis_angle)
mat2C = rctf.matrix_to_axis_angle(mat2B)

test_match( mat1A, mat2A )
test_match( mat1B, mat2B )
test_match( mat1C, mat2C )
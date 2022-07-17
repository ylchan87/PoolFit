import pickle
import cv2
import numpy as np
import tensorflow as tf
from camera import Camera
from common import SideID
from utils import *

def gen_rect(recth, rectw):
        A = tf.constant(([
                            [-0.5,  0.5, 0],
                            [ 0.5,  0.5, 0],
                            [ 0.5, -0.5, 0],
                            [-0.5, -0.5, 0],
                        ]), dtype=tf.float32) 

        B = tf.stack([rectw, recth , 1.],axis=0)
        pts = A*B

        return pts

def gen_rect2(rectRatioSqrt):
        A = tf.constant(([
                            [-0.5,  0.5, 0],
                            [ 0.5,  0.5, 0],
                            [ 0.5, -0.5, 0],
                            [-0.5, -0.5, 0],
                        ]), dtype=tf.float32) 

        B = tf.stack([rectRatioSqrt, 1./rectRatioSqrt , 1.],axis=0)
        pts = A*B

        return pts

def get_distance_from_line(lineStart, lineEnd, pts):
    """
    +ive  = on left of the line
    """
    line = -lineStart+lineEnd
    lineLength = tf.norm(line)

    lineUvec = line / lineLength #unit vec parallel to line
    lineVvec = tf.linalg.matvec( tf.constant([ [0.,1.], [-1.,0.]]) , lineUvec) #unit vec perpendicular to line

    tmp = pts - lineStart
    us = tf.linalg.matvec(tmp, lineUvec)
    vs = tf.linalg.matvec(tmp, lineVvec)

    vs = tf.abs(vs)

    mask = (us<0)
    vs = tf.tensor_scatter_nd_add(vs, tf.where(mask), tf.abs(us[mask])*1.0)

    mask = (us>lineLength)
    vs = tf.tensor_scatter_nd_add(vs, tf.where(mask), (us-lineLength)[mask]*1.0)

    return vs

def get_min_dist(corners, ref_xys):
    """
    corners: 4 corners of the to be optimized rect in img
    ref_xys: pts on the perimeter on the ref rect
    """
    distances = []
    for i in range(4):
        distances.append( get_distance_from_line(corners[i], corners[(i+1)%4], ref_xys) )
    distances = tf.abs( tf.stack(distances, axis=0) )

    minDistnace, nearestEdgeIdx = tf.min(distances, axis=0)

    return minDistnace, nearestEdgeIdx

def lossFunc(ref_xys, ref_sideIds, corners_xy):
    """
    assumes corners_xys are in topLeft topRight botRight botLeft order
    """

    sideIDs = [
        SideID.TOP,
        SideID.RIGHT,
        SideID.BOT,
        SideID.LEFT,
    ]

    loss = []
    
    for i in range(4):
        mask = ref_sideIds==sideIDs[i]
        if not tf.math.reduce_any(mask): continue
        loss.append( get_distance_from_line(corners_xy[i], corners_xy[(i+1)%4],  ref_xys[mask]) )
    
    loss = tf.concat(loss, axis=0)
    loss = tf.reduce_mean(loss * loss) # L2 loss
    return loss

def ballLossFunc(ref_ball_img_xys, ref_ball_img_d, ball_img_xys, ball_img_d):
    loss_d = ref_ball_img_d - ball_img_d
    loss_d = tf.mean(loss_d*loss_d)

    loss_xys = ref_ball_img_xys - ball_img_xys
    loss_xys = tf.mean(loss_xys*loss_xys)

    loss = loss_xys + loss_d * 5.

    return loss

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_idx", default=6, type=int)
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--init_f", default=1000, type=float)
    options = parser.parse_args()

    # color for 4 side of the rect
    PALETTE = [
        (   0,   0, 255), 
        (   0, 255, 255), 
        (   0, 255,   0), 
        ( 255,   0,   0), 
    ]

    # unit m
    # rectangle to fit
    recth = 1.4
    rectw = 2.8
    #recth = 1.0
    #rectw = tf.tensor( 1.0, dtype=tf.float32, requires_grad=True)
    rectRatioSqrt = tf.Variable( 1.0, dtype=tf.float32, trainable=True)

    balld = 0.0615

    #corners_xyz = gen_rect(recth, rectw)
    corners_xyz = gen_rect2(rectRatioSqrt)

    camera = Camera(800., 1000, 1000)
    camera.set_f(options.init_f)
    camera.set_pos([5,1,5])
    camera.set_lookat([0,0,0])    

    # camera.set_f(1363.)
    camera.set_f_fixed(True)
    
    corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

    ref_answer, ref_img = read_test_case(options.test_idx)
    ref_f          = ref_answer['f']
    ref_xys        = tf.constant(ref_answer["pts"]      , dtype=tf.float32)
    ref_xys_sideID = tf.constant(ref_answer["ptsSideID"], dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    iter = 0
    for iter in range(options.max_iter):
        with tf.GradientTape() as tape:
        
            corners_xyz = gen_rect(recth, rectw)
            corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

            # minDistnace, nearestEdgeIdx = get_min_dist(corners_xy, ref_xys)
            # #loss = tf.mean(minDistnace) # L1 loss
            # loss = tf.mean(minDistnace*minDistnace) # L1 loss
            loss = lossFunc(ref_xys, ref_xys_sideID, corners_xy)
            
        if (iter%100)==0:
            print(f"iter: {iter} loss: {loss}")

        grads = tape.gradient(loss, camera.trainable_variables)
        optimizer.apply_gradients(zip(grads, camera.trainable_variables))

        
        if loss < 1: break

        if (iter%100)==0:
            canvas = np.copy(ref_img)
            drawPolygon(corners_xy, canvas, PALETTE )

            cv2.imwrite(f"./testoutput/iter{iter:04d}.png", canvas)

    
        
    print("====Fit====")
    #print(f"rectRatio: {rectRatioSqrt*rectRatioSqrt}")        
    #print(f"rectRatio: {rectRatioSqrt*rectRatioSqrt}")        
    print(f"cam f: {camera.initf * camera.f_corrfac}")
    print(f"cam pos: {camera.pos}")
    
    print("====Ref====")
    print('f'             , ref_answer['f'             ])
    print("ball_world_xys", ref_answer["ball_world_xys"])
    print("ball_img_d"    , ref_answer["ball_img_d"    ])

    pixPerM = 500
    resulth = int(recth * pixPerM)
    resultw = int(rectw * pixPerM)

    corners_xy_in_cam, _ = camera.getPixCoords(corners_xyz, tune_cam_params=False)
    corners_xy_in_cam = corners_xy_in_cam.numpy().astype(np.float32)

    corners_xy_in_result = np.array([
        [0,0], [resultw,0], [resultw,resulth], [0,resulth]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners_xy_in_cam, corners_xy_in_result)
    dst = cv2.warpPerspective(ref_img,M,(resultw,resulth))
    cv2.imwrite('./testoutput/proj.png',dst)
    






    

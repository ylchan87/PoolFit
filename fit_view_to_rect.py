import pickle
import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from camera_pytorch import Camera

def drawPolygon(pts, canvas = None, colors = [(255,0,0)], imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    if type(pts)==torch.Tensor:
        pts = pts.cpu().detach().numpy()

    nColors = len(colors)

    pts = pts.astype(int)
    for i in range(0,len(pts)-1):
        _ = cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), colors[i%nColors], 2)

    i+=1
    _ = cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), colors[i%nColors], 2)

    return canvas

def drawDots(pts, canvas = None, colors = [(255,0,0)], imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    if type(pts)==torch.Tensor:
        pts = pts.cpu().detach().numpy()

    nColors = len(colors)

    pts = pts.astype(int)
    for i in range(0,len(pts)):
        _ = cv2.circle(canvas, tuple(pts[i]), 2, colors[i%nColors], -1)

    return canvas

def read_test_case(idx, path="./testimgs/"):
    refImg = cv2.imread(f"{path}/test_{idx:02d}.jpg")
    with open(f"{path}/test_{idx:02d}.pkl", "rb") as f:
        pts = pickle.load(f)
    return pts, refImg

def gen_rect(recth, rectw):
        pts = torch.tensor([
                [-0.5,  0.5, 0],
                [ 0.5,  0.5, 0],
                [ 0.5, -0.5, 0],
                [-0.5, -0.5, 0],
            ], dtype=torch.float32)

        pts[:,0]*=rectw
        pts[:,1]*=recth

        return pts

def gen_rect2(rectRatioSqrt):
        pts = torch.tensor([
                [-0.5,  0.5, 0],
                [ 0.5,  0.5, 0],
                [ 0.5, -0.5, 0],
                [-0.5, -0.5, 0],
            ], dtype=torch.float32)

        pts[:,0]*=rectRatioSqrt
        pts[:,1]/=rectRatioSqrt

        return pts

def get_distance_from_line(lineStart, lineEnd, pts):
    """
    +ive  = on left of the line
    """
    line = -lineStart+lineEnd
    lineLength = torch.norm(line)

    lineUvec = line / lineLength #unit vec parallel to line
    lineVvec = torch.tensor([ [0.,1.], [-1.,0.]]) @ lineUvec #unit vec perpendicular to line

    tmp = pts - lineStart
    us = lineUvec @ tmp.T
    vs = lineVvec @ tmp.T

    #try 1
    # vs[us<0] = torch.finfo(torch.float32).max
    # return vs

    #try2
    vs = torch.abs(vs)

    mask = (us<0)
    vs[mask] += torch.abs(us[mask])*1.5

    mask = (us>lineLength)
    vs[mask] += (us-lineLength)[mask]*1.5
    return vs

def get_min_dist(corners, ref_xys):
    """
    corners: 4 corners of the to be optimized rect in img
    ref_xys: pts on the perimeter on the ref rect
    """
    distances = []
    for i in range(4):
        distances.append( get_distance_from_line(corners[i], corners[(i+1)%4], ref_xys) )
    distances = torch.abs( torch.stack(distances, axis=0) )

    minDistnace, nearestEdgeIdx = torch.min(distances, axis=0)

    return minDistnace, nearestEdgeIdx

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_idx", default=0, type=int)
    parser.add_argument("--max_iter", default=10000, type=int)
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
    recth = 1.0
    rectw = torch.tensor( 1.0, dtype=torch.float32, requires_grad=True)
    #rectRatioSqrt = torch.tensor( 1.0, dtype=torch.float32, requires_grad=True)

    balld = 0.0615

    corners_xyz = gen_rect(recth, rectw)
    #corners_xyz = gen_rect2(rectRatioSqrt)

    camera = Camera(800., 1000, 1000)
    camera.set_f(1000.)
    camera.set_pos([-5,1,5])
    camera.set_lookat([0,0,0])    

    # camera.set_f(1363.)
    camera.set_f_fixed(True)
    
    corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

    ref_answer, ref_img = read_test_case(options.test_idx)

    ref_xys = torch.from_numpy(ref_answer["pts"]).to(torch.float32)

    #optimizer = optim.Adam( list(camera.parameters()) + [rectRatioSqrt], lr=0.1)
    optimizer = optim.Adam( list(camera.parameters()) + [rectw], lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    lossFunc = nn.L1Loss()

    f_fixed = True

    for iter in range(options.max_iter):

        
        corners_xyz = gen_rect(recth, rectw)
        #corners_xyz = gen_rect2(rectRatioSqrt)
        corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

        minDistnace, nearestEdgeIdx = get_min_dist(corners_xy, ref_xys)
        #loss = torch.mean(minDistnace) # L1 loss
        loss = torch.mean(minDistnace*minDistnace) # L1 loss
        

        if (iter%100)==0:
            print(f"iter: {iter} loss: {loss}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step(loss)

        if loss<100 and f_fixed:
            print("release f")
            camera.set_f_fixed(False)
            f_fixed = False

        if loss < 0.01: break

        if (iter%100)==0:
            colors = [PALETTE[i] for i in nearestEdgeIdx]
            canvas = np.zeros((1000,1000,3), dtype=np.uint8)
            drawDots(   ref_xys, canvas, colors )
            drawPolygon(corners_xy, canvas, PALETTE )

            cv2.imwrite(f"./testoutput/iter{iter:04d}.png", canvas)
        


    print("====Fit====")
    print(f"rectRatio: {rectRatioSqrt*rectRatioSqrt}")        
    print(f"cam f: {camera.initf * camera.f_corrfac}")
    print(f"cam pos: {camera.pos}")

    print("====Ref====")
    print(ref_answer['f'])





    

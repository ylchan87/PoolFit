from copyreg import pickle
import cv2
import numpy as np

from camera_pytorch import Camera
from utils import *

def genDottedLine(pt1, pt2, n=10):    
    delta = (-pt1+pt2)/n
    pts = [pt1]
    for i in range(n):
        pts.append( pt1 + i*delta)
    return np.array(pts)

def genDottedPolygon(pts):
    lines = []
    for i in range(0,len(pts)-1):
        lines.append( genDottedLine(pts[i], pts[i+1]) )
    lines.append( genDottedLine(pts[-1], pts[0]) )
    output = np.concatenate(lines, axis=0)
    return output

def randPtInBox(boxCenter, boxSize):
    if type(boxSize)==list  : boxSize   = np.array(boxSize)
    if type(boxCenter)==list: boxCenter = np.array(boxCenter)
    return np.random.random(3)*boxSize/2. + boxCenter

if __name__=="__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=False, action="store_true", help="Gen samples where 2 out of 4 lines are parallel")
    parser.add_argument("--startIdx", default=10, type=int)
    parser.add_argument("--n", default=3, type=int)
    options = parser.parse_args()

    # color for 4 side of the rect
    PALETTE = [
        (   0,   0, 255), 
        (   0, 255, 255), 
        (   0, 255,   0), 
        ( 255,   0,   0), 
    ]

    imgH = 1000
    imgW = 1000
    #unit m
    rectw = 2.8
    recth = 1.4
    balld = 0.0615

    ptl = ( -rectw/2,  recth/2, 0)
    ptr = (  rectw/2,  recth/2, 0)
    pbl = ( -rectw/2, -recth/2, 0)
    pbr = (  rectw/2, -recth/2, 0)
    pts = np.array([ptl, ptr, pbr, pbl])

    xyzs = genDottedPolygon(pts)
    #xyzs = pts

    camera = Camera(800., imgH, imgW)
    
    for idx in range(options.startIdx,options.startIdx+options.n):

        camera.set_f( np.random.randint(600,1200) )
        
        randPos    = randPtInBox([0,0,2.2], [5,5,4])
        randLookAt = randPtInBox([0,0,1], [1,1,1])

        if options.parallel:
            randPos[0] = 0.
            randLookAt[0] = 0.

        camera.set_pos( randPos )
        camera.set_lookat(randLookAt)

        xys_set, mask = camera.getPixCoords(xyzs)
        xys_set = xys_set.detach().cpu().numpy()

        xys_seti = np.round(xys_set).astype(int)
        mask = np.stack([
            0 <= xys_seti[:,0]         ,
                 xys_seti[:,0] < imgW  ,
            0 <= xys_seti[:,1]         ,
                 xys_seti[:,1] < imgH
        ])
        mask = np.all(mask, axis=0)

        xys_set  = xys_set[mask]
        xys_seti = xys_seti[mask]

        canvas = drawDots(xys_seti)
        canvas = np.zeros((1000,1000,3), dtype=np.uint8)
        drawDots(   xys_seti, canvas, PALETTE )
        #drawPolygon(xys_seti, canvas, PALETTE )

        cv2.imwrite(f"./testimgs/test_{idx:02d}.jpg", canvas) 

        with open(f"./testimgs/test_{idx:02d}.pkl", "wb") as outFile:
            pickle.dump(
                {
                    "f" : camera.f,
                    "intrinscM" : camera.intrinsicM,
                    "extrinscM" : camera.extrinsicM,
                    "pos"       : camera.pos,
                    "pts"       : xys_set

                },
                outFile)

        



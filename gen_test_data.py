from copyreg import pickle
from re import S
import cv2
import numpy as np

from camera_pytorch import Camera
from utils import *
from common import *
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

def isInCanvas(pt, h, w):
    eps = 0.01
    return (-eps<=pt[0]<(w+eps) and  -eps<=pt[1]<(h+eps))

def getIntersect(pt1, pt2, XorY, c):
    """
    XorY: 0 means X, 1 means Y

    eg.
    XorY = 0, c= w
    c : the line to intersect is X=w
    """

    dim = XorY
    dl = pt2-pt1

    neededDiff = c - pt1[dim]
    diffPerStep = dl[dim]
    
    ratio = neededDiff/diffPerStep
    if (0.<ratio<1.):
        return pt1 + dl * ratio
    else:
        return None

def clipLineToCanvas(pt1, pt2, h, w):

    output = []

    isPt1In = isInCanvas(pt1, h, w)
    isPt2In = isInCanvas(pt2, h, w)

    if isPt1In: output.append(pt1)

    if not(isPt1In and isPt2In):

        X = 0
        Y = 1

        # top intersect
        intersect = getIntersect(pt1, pt2, Y, 0)
        if intersect is not None and isInCanvas(intersect, h, w): output.append(intersect)

        # bot intersect
        intersect = getIntersect(pt1, pt2, Y, h)
        if intersect is not None and isInCanvas(intersect, h, w): output.append(intersect)

        # left intersect
        intersect = getIntersect(pt1, pt2, X, 0)
        if intersect is not None and isInCanvas(intersect, h, w): output.append(intersect)

        # right intersect
        intersect = getIntersect(pt1, pt2, X, w)
        if intersect is not None and isInCanvas(intersect, h, w): output.append(intersect)

    if isPt2In: output.append(pt2)

    assert len(output) in [0,2]
    
    return output

if __name__=="__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=False, action="store_true", help="Gen samples where 2 out of 4 lines are parallel")
    parser.add_argument("--startIdx", default=20, type=int)
    parser.add_argument("--n", default=10, type=int)
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

    #xyzs = genDottedPolygon(pts)
    xyzs = pts

    camera = Camera(800., imgH, imgW)
    
    for idx in range(options.startIdx,options.startIdx+options.n):

        camera.set_f( np.random.randint(400,800) )
        
        randPos    = randPtInBox([0,0,0.8], [2,2,0.7])
        randLookAt = randPtInBox([0,0,0], [1,1,0.3])

        #randPos = [0., -4., 4.]
        #randLookAt = [0.,0.,0.]

        if options.parallel:
            randPos[0] = 0.
            randLookAt[0] = 0.

        camera.set_pos( randPos )
        camera.set_lookat(randLookAt)

        xys_set, mask = camera.getPixCoords(xyzs)
        xys_set = xys_set.detach().cpu().numpy()

        nballs = 1
        ball_world_xyzs = np.random.random( (nballs,3) )-0.5
        ball_world_xyzs[:,0] *= rectw
        ball_world_xyzs[:,1] *= recth
        ball_world_xyzs[:,2]  = 0.0

        ball_img_xys, mask, ball_scales = camera.getPixCoords(ball_world_xyzs, getScales=True)
        ball_img_xys = ball_img_xys.detach().cpu().numpy()
        ball_scales = ball_scales.detach().cpu().numpy()

        sideIDs = [
            SideID.TOP,
            SideID.RIGHT,
            SideID.BOT,
            SideID.LEFT,
        ]

        anchorsPos  = []
        anchorsType = []
        canvas = np.zeros((1000,1000,3), dtype=np.uint8)

        for i in range(4):
            pts = clipLineToCanvas( xys_set[i], xys_set[(i+1)%4], imgH, imgW)
            for pt in pts:
                anchorsPos.append(pt)
                anchorsType.append(sideIDs[i])

            if len(pts)>0:
                ptsi = np.round(pts).astype(int)
                drawArrow( ptsi[0], ptsi[1], canvas, PALETTE[i] )
        
        anchorsPos  = np.array(anchorsPos )
        anchorsType = np.array(anchorsType)

        ball_img_xysi = ball_img_xys.astype(int)
        drawBall(ball_img_xysi[0], int(ball_scales[0]*balld/2), canvas, (255,255,255))

        cv2.imwrite(f"./testimgs/test_{idx:03d}.jpg", canvas) 

        with open(f"./testimgs/test_{idx:03d}.pkl", "wb") as outFile:
            pickle.dump(
                {
                    "f" : camera.f,
                    "intrinscM" : camera.intrinsicM,
                    "extrinscM" : camera.extrinsicM,
                    "pos"       : camera.pos,
                    "pts"       : anchorsPos,
                    "ptsSideID" : anchorsType,
                    "ballxys"   : ball_world_xyzs,
                    "ballscales": ball_scales,

                },
                outFile)

        



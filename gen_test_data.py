from copyreg import pickle
import cv2
import numpy as np

from camera_pytorch import Camera


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

def drawDots(pts, imgsize=(1000,1000)):
    canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    for pt in pts:
        _ = cv2.circle(canvas, tuple(pt), 2, (0,  255,  0), -1)
    return canvas

def randPtInBox(boxCenter, boxSize):
    if type(boxSize)==list  : boxSize   = np.array(boxSize)
    if type(boxCenter)==list: boxCenter = np.array(boxCenter)
    return np.random.random(3)*boxSize/2. + boxCenter

if __name__=="__main__":
    import pickle

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

    camera = Camera(800., imgH, imgW)
    
    for idx in range(3):

        camera.set_f( np.random.randint(600,1600) )
        camera.set_pos( randPtInBox([0,0,2.2], [5,5,4]) )
        camera.set_lookat(randPtInBox([0,0,0], [1,1,1]))
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

        



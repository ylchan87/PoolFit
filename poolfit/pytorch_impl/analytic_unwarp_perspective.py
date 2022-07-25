
import math
import cv2
import numpy as np

def perspectiveTransformRect(img, rectCorners, autoFlip=True, debug=False):
    """
    Transform an object, known to be a rectangle (eg. A4 paper), from the perspective view in the img, to a front view

    img: 
        np array of shape H,W,3 or H,W

    rectCorners:
        position of the 4 corners of the rect in the img
        array of shape 4x2  eg. [(x,y), (x,y), (x,y), (x,y)]

        expected order
        [TopLeft, TopRight, BotLeft, BotRight]

        (x,y) at (0,0) is topLeft most of the img

    autoFlip:
        if true, auto rearrange the corners to [TopLeft, TopRight, BotLeft, BotRight] order

    # code from:
    # https://stackoverflow.com/questions/38285229/calculating-aspect-ratio-of-perspective-transform-destination-image

    # algo ref:
    # https://www.microsoft.com/en-us/research/uploads/prod/2016/11/Digital-Signal-Processing.pdf

    """

    (rows,cols,_) = img.shape

    #image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    if autoFlip:
        
        rectCorners = np.array(rectCorners)
        idx = np.argmin( np.sum(rectCorners, axis=1) )
        topLeft = rectCorners[idx]

        xys = rectCorners - topLeft - 1
        angles = np.arctan2(xys[:,0], xys[:,1])
        anticlockwise_order = idxes = np.argsort(angles) # anticlockwise_order of corners starting from topLeft
        print(idxes)

        p = rectCorners[[
            anticlockwise_order[0],
            anticlockwise_order[3],
            anticlockwise_order[1],
            anticlockwise_order[2]
            ]]

    else:
        p = rectCorners

    #widths and heights of the projected image
    w1 = np.linalg.norm(p[0]-p[1])
    w2 = np.linalg.norm(p[2]-p[3])

    h1 = np.linalg.norm(p[0]-p[2])
    h2 = np.linalg.norm(p[1]-p[3])

    w = max(w1,w2)
    h = max(h1,h2)

    #visible aspect ratio
    ar_vis = float(w)/float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
    m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
    m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
    m4 = np.array((p[3][0],p[3][1],1)).astype('float32')

    #calculate the focal disrance
    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    if debug: print(f"k2 {k2} k3 {k3}")

        
    f = math.sqrt(np.abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

    if f<1e6:

        A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        #calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))
    else:
        # f hits singularity in case we look at the rect straight on, see Eq31 of ref pdf
        f = -1
        ar_real = np.sqrt( (n21*n21+n22*n22) / (n31*n31+n32*n32) )

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(p).astype('float32')
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])

    #project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(W,H))

    return dst, M, f

if __name__=="__main__":
    import argparse
    from utils import read_test_case

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_idx", default=108, type=int)
    options = parser.parse_args()

    d, img = read_test_case(options.test_idx)
    pts = d["pts"][::2] # 8 ends of the 4 lines -> 4 corners

    dst, M, solved_f = perspectiveTransformRect(img, pts)


    print("ref f :", d["f"])
    print("solved f :", solved_f)
    print("final aspect ratio :", dst.shape[0],dst.shape[1])

    cv2.imwrite('./testoutput/orig.png',img)
    cv2.imwrite('./testoutput/proj.png',dst)

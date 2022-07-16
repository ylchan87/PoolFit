import cv2
import numpy as np
import torch
import tensorflow as tf
import pickle 

def to_numpy(input):
    if type(input) == np.ndarray:
        return input
    if type(input)==torch.Tensor:
        return input.cpu().detach().numpy()    
    return input.numpy()    
    

def drawPolygon(pts, canvas = None, colors = [(255,0,0)], imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    pts = to_numpy(pts)

    nColors = len(colors)

    pts = pts.astype(int)
    i = 0
    for i in range(0,len(pts)-1):
        _ = cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), colors[i%nColors], 2)

    i+=1
    _ = cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), colors[i%nColors], 2)

    return canvas

def drawDots(pts, canvas = None, colors = [(255,0,0)], imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    pts = to_numpy(pts)

    nColors = len(colors)

    pts = pts.astype(int)
    for i in range(0,len(pts)):
        _ = cv2.circle(canvas, tuple(pts[i]), 2, colors[i%nColors], -1)

    return canvas

def drawArrow(pt1, pt2, canvas = None, color = (255,0,0), imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)

    pt1 = to_numpy(pt1)
    pt2 = to_numpy(pt2)

    _ = cv2.circle(canvas, tuple(pt1), 5, color, -1)
    _ = cv2.arrowedLine(canvas, tuple(pt1), tuple(pt2), color, 2)

    return canvas

def drawBall(pt, r, canvas = None, color = (255,0,0), imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    
    pt = to_numpy(pt)
    r  = to_numpy(r)

    _ = cv2.circle(canvas, tuple(pt), r, color,  1)
    _ = cv2.circle(canvas, tuple(pt), 2, color, -1)

    return canvas

def read_test_case(idx, path="./testimgs/"):
    refImg = cv2.imread(f"{path}/test_{idx:03d}.jpg")
    with open(f"{path}/test_{idx:03d}.pkl", "rb") as f:
        pts = pickle.load(f)
    return pts, refImg

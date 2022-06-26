import cv2
import numpy as np
import torch
import pickle 

def drawPolygon(pts, canvas = None, colors = [(255,0,0)], imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    if type(pts)==torch.Tensor:
        pts = pts.cpu().detach().numpy()

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

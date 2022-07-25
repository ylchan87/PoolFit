import pickle
import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from camera_pytorch import Camera
from common import SideID
from utils import *

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

def get_distance_from_line_old(lineStart, lineEnd, pts):
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

    vs = torch.abs(vs)

    mask = (us<0)
    vs[mask] += torch.abs(us[mask])*1.0

    mask = (us>lineLength)
    vs[mask] += (us-lineLength)[mask]*1.0

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
        if not torch.any(mask) : continue
        if ref_xys[i*2,0]<0 : continue
        loss.append( get_distance_from_line(corners_xy[i], corners_xy[(i+1)%4],  ref_xys[mask]) )
    
    loss = torch.cat(loss)
    loss = torch.mean(loss * loss) # L2 loss
    return loss

def ballLossFunc(ref_ball_img_xys, ref_ball_img_d, ball_img_xys, ball_img_d):
    loss_d = ref_ball_img_d - ball_img_d
    loss_d = torch.mean(loss_d*loss_d)

    loss_xys = ref_ball_img_xys - ball_img_xys
    loss_xys = torch.mean(loss_xys*loss_xys)

    loss = loss_xys + loss_d * 5.

    return loss

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_idx", default=8, type=int)
    parser.add_argument("--max_iter", default=400, type=int)
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
    #rectw = torch.tensor( 1.0, dtype=torch.float32, requires_grad=True)
    rectRatioSqrt = torch.tensor( 1.0, dtype=torch.float32, requires_grad=True)

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

    ref_xys        = torch.from_numpy(ref_answer["pts"]).to(torch.float32)
    ref_xys_sideID = torch.from_numpy(ref_answer["ptsSideID"]).to(torch.float32)

    optimizer = optim.Adam( list(camera.parameters()) , lr=0.1)
    #optimizer = optim.Adam( list(camera.parameters()) + [rectRatioSqrt], lr=0.1)
    #optimizer = optim.Adam( list(camera.parameters()) + [rectw], lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    f_fixed = True

    iter = 0
    from datetime import datetime
    t1 = datetime.now()

    for iter in range(options.max_iter):
        
        corners_xyz = gen_rect(recth, rectw)
        corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

        # minDistnace, nearestEdgeIdx = get_min_dist(corners_xy, ref_xys)
        # #loss = torch.mean(minDistnace) # L1 loss
        # loss = torch.mean(minDistnace*minDistnace) # L1 loss
        loss = lossFunc(ref_xys, ref_xys_sideID, corners_xy)
        

        if (iter%100)==0:
            print(f"iter: {iter} loss: {loss}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        # if loss<100 and f_fixed:
        #     print("release f")
        #     camera.set_f_fixed(False)
        #     f_fixed = False

        # if loss < 100: break

        if (iter%20)==0:
            canvas = np.copy(ref_img)
            drawPolygon(corners_xy, canvas, PALETTE )

            cv2.imwrite(f"./testoutput/iter{iter:04d}.png", canvas)
            # cv2.imwrite(f"./testoutput/step{iter//20:04d}.png", canvas)


    dt = datetime.now() - t1
    print(f" time used: {dt.total_seconds()}")

    print("release f, add ball")
    camera.set_f_fixed(False)
    f_fixed = False

    ref_ball_img_pos = torch.tensor(ref_answer["ball_img_xys"], dtype=torch.float32)
    ref_ball_img_d   = torch.tensor(ref_answer["ball_img_d"  ], dtype=torch.float32)
    rough_pos_guess = camera.getWorldCoords(ref_ball_img_pos)[0]
    ball_world_pos_x = torch.tensor( rough_pos_guess[0], dtype=torch.float32, requires_grad=True)
    ball_world_pos_y = torch.tensor( rough_pos_guess[1], dtype=torch.float32, requires_grad=True)
    
    ball_world_pos = torch.tensor( [0., 0., 0.] , dtype=torch.float32)
    ball_world_pos[0] = ball_world_pos_x
    ball_world_pos[1] = ball_world_pos_y


    optimizer = optim.Adam( list(camera.parameters()) + [ball_world_pos_x, ball_world_pos_y] , lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    for iter2 in range(iter, options.max_iter):
        
        corners_xyz = gen_rect(recth, rectw)
        corners_xy, mask = camera.getPixCoords(corners_xyz, tune_cam_params=True)

        edge_loss = lossFunc(ref_xys, ref_xys_sideID, corners_xy)


        ball_world_pos = torch.tensor( [0., 0., 0.] , dtype=torch.float32)
        ball_world_pos[0] = ball_world_pos_x
        ball_world_pos[1] = ball_world_pos_y

        ball_img_xys, mask, scales = camera.getPixCoords(ball_world_pos.unsqueeze(0), tune_cam_params=True, getScales=True)
        ball_img_d = scales*balld
        ball_loss = ballLossFunc(ref_ball_img_pos, ref_ball_img_d, ball_img_xys, ball_img_d)

        # ball_img_xys, mask = camera.getPixCoords(ball_world_pos.unsqueeze(0), tune_cam_params=True, getScales=False, doUpdate=False)
        # ball_loss = ballLossFunc(ref_ball_img_pos, ref_ball_img_d, ball_img_xys, ref_ball_img_d)

        loss = edge_loss + ball_loss
        #loss = edge_loss

        if (iter2%100)==0:
            print(f"iter2: {iter2} loss: {loss}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        if loss < 0.1: break

        if (iter2%100)==0:
            canvas = np.copy(ref_img)
            drawPolygon(corners_xy, canvas, PALETTE )

            ball_img_xysi = ball_img_xys.to(int)
            drawBall(ball_img_xysi[0], int(ball_img_d[0]/2), canvas, (255,255,255))

            cv2.imwrite(f"./testoutput/iter{iter2:04d}.png", canvas)
        
    print("====Fit====")
    #print(f"rectRatio: {rectRatioSqrt*rectRatioSqrt}")        
    #print(f"rectRatio: {rectRatioSqrt*rectRatioSqrt}")        
    print(f"cam f: {camera.initf * camera.f_corrfac}")
    print(f"cam pos: {camera.pos}")
    print(f"ball pos: {ball_world_pos_x} {ball_world_pos_y}")
    print(f"ball d: {ball_img_d[0]}")

    print("====Ref====")
    print('f'             , ref_answer['f'             ])
    print("ball_world_xys", ref_answer["ball_world_xys"])
    print("ball_img_d"    , ref_answer["ball_img_d"    ])

    pixPerM = 500
    resulth = int(recth * pixPerM)
    resultw = int(rectw * pixPerM)

    corners_xy_in_cam, _ = camera.getPixCoords(corners_xyz, tune_cam_params=False)
    corners_xy_in_cam = corners_xy_in_cam.detach().numpy().astype(np.float32)

    corners_xy_in_result = np.array([
        [0,0], [resultw,0], [resultw,resulth], [0,resulth]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners_xy_in_cam, corners_xy_in_result)
    dst = cv2.warpPerspective(ref_img,M,(resultw,resulth))
    cv2.imwrite('./testoutput/proj.png',dst)
    






    

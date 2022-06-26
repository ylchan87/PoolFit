import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from rotation_conversion import matrix_to_axis_angle, axis_angle_to_matrix

def normalize(v):
    norm = torch.norm(v)
    if norm==0: return v
    return v/norm

class Camera(nn.Module):
    def __init__(self, initf, imgH, imgW) -> None:
        super().__init__()

        # intrinsics related
        self.imgW  = imgW
        self.imgH  = imgH
        self.initf = initf
        self.f_corrfac  = nn.Parameter( torch.tensor(     [1.0], dtype=torch.float32) )

        # extrinsics related
        self._pos             = torch.tensor([1.,1.,1.], dtype=torch.float32) # pos in world frame
        self.pos_in_cam_frame = nn.Parameter( torch.tensor([1.,1.,1.], dtype=torch.float32) )
        self.axis_angle       = nn.Parameter( torch.tensor([0.,0.,0.], dtype=torch.float32) )

        self.lookat = torch.tensor([0.,0.,0.], dtype=torch.float32)
        self.roll   = 0.

        self.intrinsicM   = torch.eye(3, dtype=torch.float32)
        self.extrinsicM   = torch.eye(4, dtype=torch.float32)[0:3]
        self.perspectiveM = torch.eye(4, dtype=torch.float32)[0:3]

        self.fov_deg_h = -1
        self.fov_deg_v = -1

        self._update_fov()
        self._update_intrinsicM()
        self._update_extrinsicM()

    @property
    def f(self):
        return 1./self.f_corrfac.detach().item()* self.initf
        
    @f.setter
    def f(self, val):
        with torch.no_grad():
            self.initf = val
            self.f_corrfac[:] = 1.0
            self._update_fov()
            self._update_intrinsicM()
    
    @property
    def pos(self):
        return -self.extrinsicM[:3,:3].T @ self.pos_in_cam_frame
        
    @pos.setter
    def pos(self, xyz):
        with torch.no_grad():
            if type(xyz) in [list, np.ndarray]: xyz = torch.tensor(xyz, dtype=torch.float32)
            self._pos[:] = xyz
            self.set_lookat(self.lookat)
            self._update_extrinsicM()        

    def set_f(self, val):        
        self.f = val
    
    def set_img_size(self, imgH, imgW):
        with torch.no_grad():
            self.imgH = imgH
            self.imgW = imgW
            self._update_intrinsicM()

    def set_pos(self, xyz):
        self.pos = xyz

    def set_lookat(self, lookat, roll=0.):
        if roll!=0.:
            raise NotImplementedError("roll != 0 not yet work")
        
        if type(lookat) in [list, np.ndarray]: lookat = torch.tensor(lookat, dtype=torch.float32)
        self.lookat = lookat
        self.roll = roll
        
        camZ = normalize(-self._pos + self.lookat)

        if camZ[0]==camZ[1]==0:
            camX = torch.tensor([1., 0., 0.], dtype=torch.float32)
        elif abs(camZ[1])>abs(camZ[0]):
            camX = torch.tensor([1., -camZ[0]/camZ[1], 0.], dtype=torch.float32)
        else:
            camX = torch.tensor([-camZ[1]/camZ[0], 1., 0.], dtype=torch.float32)
        camX = normalize(camX)

        camY = torch.cross(camZ, camX)

        if camY[2]<0:
            camY = -camY
            camX = -camX

        rotM = torch.stack([camX, camY, camZ], axis=0)

        with torch.no_grad():
            self.pos_in_cam_frame[:] = -rotM @ self._pos 
            self.axis_angle[:] = matrix_to_axis_angle(rotM)
            self._update_extrinsicM()   

    def set_f_fixed(self, flag):
        self.f_corrfac.requires_grad = not flag

    def _update_fov(self):
        self.fov_deg_w = np.rad2deg( np.arctan(self.imgW/2/self.f)*2 )
        self.fov_deg_h = np.rad2deg( np.arctan(self.imgH/2/self.f)*2 )
        print(f"FOV (deg) Horizontal:{self.fov_deg_h} ")

    def _update_intrinsicM(self):
        """
         PixXYW = IntrinsicM @ CamFrameXYZ
         need to norm the w component of PixXYW to get pix (x,y) coord
        """
        # self.f will be casted as a value(?) in tensor construct -> NO grad
        self.intrinsicM =  torch.tensor([
                                [ -self.f,       0, self.imgW/2],
                                [       0, -self.f, self.imgH/2],
                                [       0,       0,           1],
                            ], dtype=torch.float32)
                            
        # assign self.f to elements -> has grad
        self.intrinsicM[0,0] = -1./self.f_corrfac * self.initf
        self.intrinsicM[1,1] = -1./self.f_corrfac * self.initf

        self.perspectiveM = self.intrinsicM @ self.extrinsicM
        
    def _update_extrinsicM(self):
        """
         CamFrameXYZ = extrinsicM @ WorldFrameXYZ1
        """
        extrinsicM = torch.eye(4, dtype=torch.float32)[0:3]
        rotM = axis_angle_to_matrix( self.axis_angle)
        
        extrinsicM[0:3, 0:3] = rotM
        extrinsicM[0:3,   3] = self.pos_in_cam_frame

        self.extrinsicM = extrinsicM
        self.perspectiveM = self.intrinsicM @ self.extrinsicM

    def getPixCoords(self, worldCoords, tune_cam_params=False, getScales=False):
        """
        worldCoords : numpy array (n,3)
        """
        if tune_cam_params:
            self._update_intrinsicM()
            self._update_extrinsicM()
            perspectiveM = self.perspectiveM
        else:
            perspectiveM = self.perspectiveM.detach()

        if type(worldCoords) in [np.ndarray]: worldCoords = torch.tensor(worldCoords, dtype=torch.float32)

        xyzws = torch.cat( [worldCoords, torch.ones([len(worldCoords), 1])], axis=1 )
        xyws = perspectiveM @ xyzws.T
        xyws = xyws.T
        ws = xyws[:,2]
        mask = ws>0            # if false, point is behind the cam
        xys = xyws[:,0:2] / ws[:,None] # normalize w to 1

        if getScales:
            cam_frame_xyzs = self.extrinsicM @ xyzws.T
            scale = 1./self.f_corrfac* self.initf / cam_frame_xyzs[2]
            return xys, mask, scale
        else:
            return xys, mask
        
    

def drawPolygon(pts, canvas = None, color = (255,0,0), imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    if type(pts)==torch.Tensor:
        pts = pts.cpu().detach().numpy()

    pts = pts.astype(int)
    for i in range(0,len(pts)-1):
        _ = cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), color, 2)
    _ = cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), color, 2)
    return canvas

if __name__=="__main__":
    # unit m
    # rectangle to fit
    recth = 1.0
    rectw = torch.tensor( 1.0, dtype=torch.float32, requires_grad=True)
    balld = 0.0615

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

    pts = gen_rect(recth, rectw)

    camera = Camera(800., 1000, 1000)
    camera.set_f(1000.)
    camera.set_pos([-5,1,5])
    camera.set_lookat([0,0,0])    
    
    xys, mask = camera.getPixCoords(pts, tune_cam_params=True)
    print(xys)
    print(mask)

    ref_xys = torch.tensor([
            [317.91403682 ,535.04245995],
            [548.6423495  ,415.74897926],
            [662.0134948  ,468.82048839],
            [430.72150222 ,619.99387802],
        ], dtype = torch.float32)

    opt = optim.Adam( list(camera.parameters()) + [rectw], lr=0.01)
    lossFunc = nn.L1Loss()
    for iter in range(10000):
        

        pts = gen_rect(recth, rectw)
        xys, mask = camera.getPixCoords(pts, tune_cam_params=True)
        loss = lossFunc(xys, ref_xys)
        print(f"iter: {iter} loss: {loss}")
        loss.backward()
        opt.step()
        opt.zero_grad()

        if (iter%100)==0:
            canvas = np.zeros((1000,1000,3), dtype=np.uint8)
            drawPolygon(ref_xys, canvas, (0,255,0) )
            drawPolygon(    xys, canvas, (0,0,255) )
            cv2.imwrite(f"./testoutput/iter{iter:04d}.png", canvas)

    print(f"rectw: {rectw}")        
    print(f"cam f: {camera.initf / camera.f_corrfac}")
    print(f"cam pos: {camera.pos}")




    

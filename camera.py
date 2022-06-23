import cv2
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm==0: return v
    return v/norm

class Camera:
    def __init__(self, f, imgH, imgW) -> None:
        self.f = f
        self.imgW = imgW
        self.imgH = imgH

        self.pos    = np.array([1.,1.,1.])
        self.lookat = np.array([0.,0.,0.])
        self.roll   = 0.

        self.intrinsicM = np.eye(3)
        self.extrinsicM = np.eye(4)[0:3]
        self.perspectiveM = np.eye(4)[0:3]

        self.fov_deg_h = -1
        self.fov_deg_v = -1

        self._update_intrinsicM()
        self._update_extrinsicM()

    def set_f(self, f):
        self.f = f
        self._update_intrinsicM()
    
    def set_img_size(self, imgH, imgW):
        self.imgH = imgH
        self.imgW = imgW
        self._update_intrinsicM()

    def set_pos(self, xyz):
        if type(xyz)==list: xyz = np.array(xyz)
        self.pos = xyz
        self._update_extrinsicM()        

    def set_lookat(self, pt, roll=0.):
        if roll!=0.:
            raise NotImplementedError("roll != 0 not yet work")
        if type(pt)==list: pt = np.array(pt)
        self.lookat = pt
        self.roll = roll
        self._update_extrinsicM()        

    def _update_intrinsicM(self):
        """
         PixXYW = IntrinsicM @ CamFrameXYZ
         need to norm the w component of PixXYW to get pix (x,y) coord
        """
        self.intrinsicM =  np.array([
                                [ -self.f,       0, self.imgW/2],
                                [       0, -self.f, self.imgH/2],
                                [       0,       0,           1],
                            ])
        self.perspectiveM = self.intrinsicM @ self.extrinsicM

        self.fov_deg_w = np.rad2deg( np.arctan(self.imgW/2/self.f)*2 )
        self.fov_deg_h = np.rad2deg( np.arctan(self.imgH/2/self.f)*2 )
        print(f"FOV (deg) Horizontal:{self.fov_deg_h} ")
        
        
    def _update_extrinsicM(self):
        """
         CamFrameXYZ = extrinsicM @ WorldFrameXYZ1
        """
        camZ = normalize(-self.pos + self.lookat)

        if camZ[0]==camZ[1]==0:
            camX = np.array([1., 0., 0.])
        elif abs(camZ[1])>abs(camZ[0]):
            camX = np.array([1., -camZ[0]/camZ[1], 0.])
        else:
            camX = np.array([-camZ[1]/camZ[0], 1., 0.])
        camX = normalize(camX)

        camY = np.cross(camZ, camX)

        if camY[2]<0:
            camY = -camY
            camX = -camX
        
        extrinsicM = np.eye(4)[0:3]

        extrinsicM[0, 0:3] = camX
        extrinsicM[1, 0:3] = camY
        extrinsicM[2, 0:3] = camZ

        extrinsicM[0, 3] = -camX @ self.pos
        extrinsicM[1, 3] = -camY @ self.pos
        extrinsicM[2, 3] = -camZ @ self.pos

        self.extrinsicM = extrinsicM
        self.perspectiveM = self.intrinsicM @ self.extrinsicM

    def getPixCoords(self, worldCoords):
        """
        worldCoords : numpy array (n,3)
        """
        xyzws = np.concatenate( [worldCoords, np.ones([len(worldCoords), 1])], axis=1 )
        xyws = self.perspectiveM @ xyzws.T
        xyws = xyws.T
        ws = xyws[:,2]
        mask = ws>0            # if false, point is behind the cam
        xys = xyws[:,0:2] / ws[:,None] # normalize w to 1
        return xys, mask

if __name__=="__main__":
    #unit m
    rectw = 2.8
    recth = 1.4
    balld = 0.0615

    ptl = ( -rectw/2,  recth/2, 0)
    ptr = (  rectw/2,  recth/2, 0)
    pbl = ( -rectw/2, -recth/2, 0)
    pbr = (  rectw/2, -recth/2, 0)

    camera = Camera(800., 1000, 1000)
    camera.set_f(800.)
    camera.set_pos([-4,-4,4])
    camera.set_lookat([0,0,0])

    pts = np.array([ptl, ptr, pbl, pbr, ptl, ptr, pbl, pbr])
    
    xys, mask = camera.getPixCoords(pts)
    print(xys)
    print(mask)

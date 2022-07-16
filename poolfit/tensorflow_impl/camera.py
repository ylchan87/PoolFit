from hmac import trans_36
import cv2
import numpy as np
from poolfit.tensorflow_impl.rotation_conversion import matrix_to_axis_angle, axis_angle_to_matrix
import tensorflow as tf

def normalize(v):
    norm = tf.norm(v)
    if norm==0: return v
    return v/norm

class Camera(tf.keras.Model):
    def __init__(self, initf, imgH, imgW) -> None:
        super().__init__()

        # intrinsics related
        self.imgW  = imgW
        self.imgH  = imgH
        self.initf = initf
        self.f_corrfac  = tf.Variable( 1.0, name="f_corrfac", dtype=tf.float32, trainable=True )

        # extrinsics related
        self._pos             = tf.Variable( [1.,1.,1.], name="_pos"            , dtype=tf.float32, trainable=False) # pos in world frame
        self.pos_in_cam_frame = tf.Variable( [1.,1.,1.], name="pos_in_cam_frame", dtype=tf.float32, trainable=True)
        self.axis_angle       = tf.Variable( [0.,0.,0.], name="axis_angle"      , dtype=tf.float32, trainable=True)

        self.lookat = tf.Variable([0.,0.,0.], dtype=tf.float32, trainable=False)
        self.roll   = 0.

        self.intrinsicM   = tf.eye(3, dtype=tf.float32)
        self.extrinsicM   = tf.eye(4, dtype=tf.float32)[0:3]
        self.perspectiveM = tf.eye(4, dtype=tf.float32)[0:3]

        self.fov_deg_h = -1
        self.fov_deg_v = -1

        self._update_fov()
        self._update_intrinsicM()
        self._update_extrinsicM()

    @property
    def f(self):
        return 1./self.f_corrfac* self.initf
        
    @f.setter
    def f(self, val):
        self.initf = val
        self.f_corrfac.assign(1.0)
        self._update_fov()
        self._update_intrinsicM()
    
    @property
    def pos(self):
        return tf.transpose(-self.extrinsicM[:3,:3]) @ self.pos_in_cam_frame
        
    @pos.setter
    def pos(self, xyz):
        self._pos.assign(xyz)
        self.set_lookat(self.lookat)
        self._update_extrinsicM()        

    def set_f(self, val):        
        self.f = val
    
    def set_img_size(self, imgH, imgW):
        self.imgH = imgH
        self.imgW = imgW
        self._update_intrinsicM()

    def set_pos(self, xyz):
        self.pos = xyz

    def set_lookat(self, lookat, roll=0.):
        if roll!=0.:
            raise NotImplementedError("roll != 0 not yet work")
        
        self.lookat.assign(lookat)
        self.roll = roll
        
        camZ = normalize(-self._pos + self.lookat)

        if camZ[0]==camZ[1]==0:
            camX = tf.Variable([1., 0., 0.], dtype=tf.float32, trainable=False)
        elif abs(camZ[1])>abs(camZ[0]):
            camX = tf.Variable([1., -camZ[0]/camZ[1], 0.], dtype=tf.float32, trainable=False)
        else:
            camX = tf.Variable([-camZ[1]/camZ[0], 1., 0.], dtype=tf.float32, trainable=False)
        camX = normalize(camX)

        camY = tf.linalg.cross(camZ, camX)

        if camY[2]<0:
            camY = -camY
            camX = -camX

        rotM = tf.stack([camX, camY, camZ], axis=0)

        #with tf.no_grad():
        self.pos_in_cam_frame.assign( tf.linalg.matvec(-rotM, self._pos))
        self.axis_angle.assign(matrix_to_axis_angle(rotM))
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
        self.intrinsicM =  tf.Variable([
                                [ -self.f,       0, self.imgW/2],
                                [       0, -self.f, self.imgH/2],
                                [       0,       0,           1],
                            ], dtype=tf.float32)
                            
        # assign self.f to elements -> has grad
        self.intrinsicM = tf.tensor_scatter_nd_update( 
                                self.intrinsicM, 
                                [[0,0],[1,1]],
                                [-1./self.f_corrfac * self.initf, -1./self.f_corrfac * self.initf]
                            )

        self.perspectiveM = self.intrinsicM @ self.extrinsicM
        
    def _update_extrinsicM(self):
        """
         CamFrameXYZ = extrinsicM @ WorldFrameXYZ1
        """
        extrinsicM = tf.Variable(tf.eye(4, dtype=tf.float32)[0:3], trainable=False)
        rotM = axis_angle_to_matrix( self.axis_angle)
        
        extrinsicM = tf.concat( [rotM, tf.expand_dims(self.pos_in_cam_frame,1)], axis=1) # 3x3 ,3x1 -> 3x4

        self.extrinsicM = extrinsicM
        self.perspectiveM = self.intrinsicM @ self.extrinsicM

    def getPixCoords(self, worldCoords, tune_cam_params=False, getScales=False, doUpdate=True):
        """
        worldCoords : numpy array (n,3)
        """
        if tune_cam_params:
            if doUpdate:
                self._update_intrinsicM()
                self._update_extrinsicM()
            perspectiveM = self.perspectiveM
        else:
            perspectiveM = self.perspectiveM.detach()

        if type(worldCoords) in [np.ndarray]: worldCoords = tf.Variable(worldCoords, dtype=tf.float32)

        xyzws = tf.concat( [worldCoords, tf.ones([worldCoords.shape[0], 1])], axis=1 )
        xyws = perspectiveM @ tf.transpose(xyzws)
        xyws = tf.transpose(xyws)
        ws = xyws[:,2]
        mask = ws>0            # if false, point is behind the cam
        xys = xyws[:,0:2] / ws[:,None] # normalize w to 1

        if getScales:
            cam_frame_xyzs = self.extrinsicM @ tf.transpose(xyzws)
            scale = 1./self.f_corrfac* self.initf / cam_frame_xyzs[2]
            return xys, mask, scale
        else:
            return xys, mask
    
    def getWorldCoords(self, imgCoords):
        """
        inf possible points along a ray in world can give same XY in img, we give the pt that is on the z=0 plane at the world
        """
        #with tf.no_grad():
        if type(imgCoords) in [np.ndarray]: imgCoords = tf.Variable(imgCoords, dtype=tf.float32)
        imgCoords = tf.cat( [imgCoords, tf.ones([len(imgCoords), 1])], axis=1 )
        invM = tf.linalg.inv(self.perspectiveM.detach()[:,[0,1,3]])
        world_xyws = invM @ tf.transpose(imgCoords)
        world_xyws = tf.transpose(world_xyws)

        world_xyzs = tf.zeros_like(world_xyws)
        world_xyzs[:,0:2] = world_xyws[:,0:2] / world_xyws[:,2]
        
        return world_xyzs

def drawPolygon(pts, canvas = None, color = (255,0,0), imgsize=(1000,1000)):
    if canvas is None:
        canvas = np.zeros((imgsize[0],imgsize[1],3), dtype=np.uint8)
    if type(pts)==tf.Variable:
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
    rectw = tf.Variable( 1.0, dtype=tf.float32, trainable=True)
    balld = 0.0615

    def gen_rect(recth, rectw):
        pts = tf.Variable([
                [-0.5*rectw,  0.5*recth, 0],
                [ 0.5*rectw,  0.5*recth, 0],
                [ 0.5*rectw, -0.5*recth, 0],
                [-0.5*rectw, -0.5*recth, 0],
            ], dtype=tf.float32)

        # pts[:,0]*=rectw
        # pts[:,1]*=recth

        return pts

    pts = gen_rect(recth, rectw)

    camera = Camera(800., 1000, 1000)
    camera.set_f(1000.)
    camera.set_pos([-5,1,5])
    camera.set_lookat([0,0,0])    
    
    xys, mask = camera.getPixCoords(pts, tune_cam_params=True)
    print(xys)
    print(mask)

    ref_xys = tf.Variable([
            [317.91403682 ,535.04245995],
            [548.6423495  ,415.74897926],
            [662.0134948  ,468.82048839],
            [430.72150222 ,619.99387802],
        ], dtype = tf.float32)

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)

    for i in range(3):

        with tf.GradientTape() as tape:
            xys, mask = camera.getPixCoords(pts, tune_cam_params=True)
            loss = tf.nn.l2_loss(ref_xys - xys)
        print(i, loss)
        var_list = camera.trainable_variables + [rectw]
        grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))



    # opt = optim.Adam( list(camera.parameters()) + [rectw], lr=0.01)
    # lossFunc = nn.L1Loss()
    # for iter in range(10000):
        

    #     pts = gen_rect(recth, rectw)
    #     xys, mask = camera.getPixCoords(pts, tune_cam_params=True)
    #     loss = lossFunc(xys, ref_xys)
    #     print(f"iter: {iter} loss: {loss}")
    #     loss.backward()
    #     opt.step()
    #     opt.zero_grad()

    #     if (iter%100)==0:
    #         canvas = np.zeros((1000,1000,3), dtype=np.uint8)
    #         drawPolygon(ref_xys, canvas, (0,255,0) )
    #         drawPolygon(    xys, canvas, (0,0,255) )
    #         cv2.imwrite(f"./testoutput/iter{iter:04d}.png", canvas)

    # print(f"rectw: {rectw}")        
    # print(f"cam f: {camera.initf / camera.f_corrfac}")
    # print(f"cam pos: {camera.pos}")




    

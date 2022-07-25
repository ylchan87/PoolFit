# About
Self study on using pytorch/tensorflow autodiff to solve an optimization problem

# Problem
Given arbitary image of a pool table, get the top down view of the table:

The problem is raised in:
https://stackoverflow.com/questions/29181863/getperspectivetransform-on-a-non-entire-quadrangle

With 4 corners visible, the problem is well defined with solution as in:
https://stackoverflow.com/questions/38285229/calculating-aspect-ratio-of-perspective-transform-destination-image

The problem is if this is still doable with 3 side only:

# Approach

The top down view and the photo view is related by the homograpgy transform matrix `M`, which has 9 elements. Luckily not all 9 are independent. 

From https://www.cse.unr.edu/~bebis/CS791E/Notes/CameraParameters.pdf, we could see `M` can be formed by knowing 
- x,y,z of the camera
- 3 rotation angles of the camera (eg. pitch roll yaw)
- magnification of the camera fx, fy 

We further assume `fx=fy` and that leave us with 7 variable.

Note that x, y, z, pitch, roll, yaw change in every shot, but f is intrinsic property of the camera thus remain unchanged across all shots taken with the same camera.


With 4 corners there's analytic solution. We can solve `M` as well as focal length `f` of the cam . The code is at `poolfit/pytorch_impl/analytic_unwarp_perspective.py`

With 3 sides we have 3x2=6 contraints (eg. slope and intercept of each side)

Asuume `f` is already known (eg. by taking image of the table with 4 corners beforehand), we can then solve the 6 variables with the 6 constraints.

Maybe there's analytic solution, but I went for fitting with gradient descent.


# How to Run

```
cd pytorch_impl
python fit_view_to_rect.py   
```
or
```
cd tensorflow_impl
python fit_view_to_rect.py   
```

outputs are saved at folder `testoutput`

# Env setup
`conda env create -f environment.yml`

or 

```
conda create --name poolfit
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install opencv-contrib-python
pip install tensorflow
```

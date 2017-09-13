### Performs image stitching using rotations from VICON ground truth and camera images ###
### Written by Varun Gupta. Dated 21st March, 2017 ###

# Change the dataests to be loaded in lines 13 and 17
# Create the appropriate folders according to line 68

from __future__ import division
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import cv2

vicon = io.loadmat('vicon/viconRot8.mat')
ts_vicon = vicon['ts']
print np.shape(ts_vicon)

cam = io.loadmat('cam/cam8.mat')
print np.shape(cam['cam'])

ts_cam = cam['ts']
print np.shape(ts_cam)

num_frames = np.shape(ts_cam)[1]
# fig = plt.figure()

f = 270
img = cam['cam'][:,:,:,0]
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

im_size = np.shape(img)
print im_size
[X,Y] = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
Xvec = np.reshape(X, (im_size[0]*im_size[1], 1))
Yvec = np.reshape(Y, (im_size[0]*im_size[1], 1))
Xc = Xvec - im_size[1]/2
Yc = Yvec - im_size[0]/2
BP = np.vstack((Xc.T, Yc.T, f*np.ones((im_size[0]*im_size[1]))))

frame = np.zeros((1400, 1580, 3))

frameNum = 1

plt.ion()

for i in xrange(200, num_frames-200, 3):
	print i
	plt.clf()
	ind = np.argmin(np.abs(ts_vicon-ts_cam[0,i]))
	R = np.squeeze(vicon['rots'][:,:,ind])
	img = np.squeeze(cam['cam'][:,:,:,i])
	img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	BPd = np.matmul(np.array([[0,0,1],[-1,0,0],[0,-1,0]]), BP)
	WP = np.matmul(R, BPd)
	theta = np.arctan2(WP[1,:], WP[0,:])
	h = np.divide(WP[2,:], np.sqrt(np.power(WP[0,:],2) + np.power(WP[1,:],2)))
	t_coord = np.int_(np.vstack((-f*theta, f*h)) + np.tile(np.vstack((np.shape(frame)[0:2]))/2, im_size[0]*im_size[1]))
	t_coord[t_coord<0] = 1
	# print t_coord

	for k in xrange(0, im_size[0]*im_size[1]):
		if (t_coord[1,k] >= np.shape(frame)[0]) | (t_coord[0,k] >= np.shape(frame)[1]):
			continue
		else:
			# print img[Yvec[k,0], Xvec[k,0], :]
			# print np.ceil(t_coord[1,k]), np.ceil(t_coord[0,k])
			frame[t_coord[1,k], t_coord[0,k], :] = img[Yvec[k,0], Xvec[k,0], :]
	plt.imshow(np.flipud(frame))
	fname = "output/data8/%05d.png"%frameNum
	fig1 = plt.gcf()
	plt.show()
	plt.draw()
	fig1.savefig(fname)
	frameNum += 1
	plt.pause(0.01)




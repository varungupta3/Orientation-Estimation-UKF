import numpy as np
from transformations import euler_from_matrix, euler_from_quaternion
from scipy import io
import matplotlib.pyplot as plt

x_hat = np.load('x_hat_imu4_update.npy')
vicon = io.loadmat("vicon/viconRot4.mat")

n = np.shape(x_hat)[1]

r_hat = []
p_hat = []
y_hat = []

ts1 = []

for i in range(0, n):
	ts1.append(x_hat[0,i])
	r, p, y = euler_from_quaternion(x_hat[1:5,i])
	r_hat.append(r)
	p_hat.append(p)
	y_hat.append(y)

ts2 = vicon['ts']
num_size = np.shape(ts2)[1]

ts2 = []
r_true = []
p_true = []
y_true = []
for i in range(0, num_size):
	R = vicon['rots'][:,:,i]
	ts2.append(vicon['ts'][0,i])
	r, p, y = euler_from_matrix(R)
	r_true.append(r)
	p_true.append(p)
	y_true.append(y)
	# print r, p, y

plt.figure(1)

plt.subplot(311)
plt.plot(ts1, r_hat, 'r')
plt.plot(ts2, r_true, 'b')
plt.title('Roll')

plt.subplot(312)
plt.plot(ts1, p_hat, 'r')
plt.plot(ts2, p_true, 'b')
plt.title('Pitch')

plt.subplot(313)
plt.plot(ts1, y_hat, 'r')
plt.plot(ts2, y_true, 'b')
plt.title('Yaw')

plt.show()
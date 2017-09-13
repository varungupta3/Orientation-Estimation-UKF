from __future__ import division
from scipy import io
import numpy as np
from quaternion_utils import Quaternions

imu = io.loadmat("imu/imuRaw4.mat")
filename = 'x_hat_imu4_update'

# Reading IMU data 
ts = np.float64(np.transpose(imu['ts']))
ts_curr = ts[1:]
ts_prev = ts[:-1]
ts_diff = np.append(0, ts_curr - ts_prev) # Time difference between subsequent measurements

bias_acc = np.array([[511],[501],[503]])
bias_gyro = np.array([[370],[374],[376]])

acc = ((np.int16(imu['vals'][0:3,:]) - bias_acc)*(3300/1023.0))/330 # converting to accelerations (-x, -y, z) in terms of g
acc[0:2,:]=-acc[0:2,:] # converting to accelerations (x, y, z) in terms of g
gyro = ((np.int16(imu['vals'][3:6,:])- bias_gyro)*(3300/1023.0)*np.pi/180)/3.33 # converting to yaw, roll, pitch
gyro = np.vstack((gyro[1:3,:], gyro[0,:])) # stacking in the order : roll, pitch, yaw

total_time = np.shape(ts)[0]

# Initializing parameters
n = 3 # Number of state variables
P_prev = 0.1*np.eye(n, dtype=float) # Error covariance of previous estimate of the state
Q = 1*np.eye(n, dtype=float) # Process noise covariance. A tuned parameter.
R = 18*np.eye(n, dtype=float) # Measurement noise covariance. A tuned parameter.
g = Quaternions(0., np.array([0., 0., 1.])) # Gravity vector as a quaternion

x_hat_prev = np.array([1.0, 0.0, 0.0, 0.0]) # Initialize the state estimate
q_mean = Quaternions(x_hat_prev[0], x_hat_prev[1:]) # Starting point for quaternion averaging in the prediction step

x_pred_hat = np.zeros((4, total_time)) # State estimates after the prediction step
x_hat = np.zeros((4, total_time)) # State estimate after applying the Kalman filter

# Constant parameters
coeff = np.sqrt(2*n) # Coefficient for sigma points
weights = np.linspace(1/(4.*n), 1/(4.*n), num=2*n) # Weights for quaternion averaging

x_test_prev = np.array([1.0, 0.0, 0.0, 0.0]) # State estimates using control (based on gyro data) only
x_test_hat = np.zeros((4, total_time))

for t in range(0, total_time):
	print 'time instant = ', t

	# Control variables using gyro data
	omega_curr = gyro[:,t]
	ang_vel = np.linalg.norm(omega_curr)
	del_angle = ang_vel * (ts_diff[t])
	if ang_vel == 0:
		del_axis = np.array([0., 0., 0.])
	else:
		del_axis = omega_curr/ang_vel
	del_q = Quaternions(np.cos(del_angle/2), np.sin(del_angle/2) * del_axis) # Change in orientation due to control input

	# Sigma Point Computation
	S = np.linalg.cholesky(P_prev + Q)
	W_i = np.hstack((coeff * S, -coeff * S)) # 2n sigma points
	
	q_prev = Quaternions(x_hat_prev[0], x_hat_prev[1:]) # Quaternion of the previous state orientation estimate

	X_i = np.empty((4, 2*n)) # Sigma Points
	Y_i = np.empty((4, 2*n)) # Transformed Sigma Points

	vector_of_quat = np.empty((2*n, 1), dtype=object) # Quaternions stored in a vector

	for i in range(0, 2*n):
		q_Wi = Quaternions(0., W_i[:,i]).exp() # Converting noise rotation vector to quaternion by exponentiating
		q_curr = q_prev * q_Wi # Adding noise to the previous state
		X_i[:,i] = np.append(q_curr.qs, q_curr.qv)
		q_curr = q_curr * del_q # Applying process model to the previous state
		Y_i[:,i] = np.append(q_curr.qs, q_curr.qv) # Transformed sigma points on applying process model
		vector_of_quat[i] = q_curr

	# Computing the mean of the transformed sigma points
	q_mean, W_idash = q_mean.average(vector_of_quat, weights) # Uses the previous state as the starting point for the iteration
	apriori_x_curr = np.append(q_mean.qs, q_mean.qv) # 

	# Testing without prediction. Just the control
	q_x_test = Quaternions(x_test_prev[0], x_test_prev[1:])
	q_x_test = q_x_test * del_q
	x_test_prev = np.append(q_x_test.qs, q_x_test.qv)
	x_test_hat[:,t] = x_test_prev

	# Computing the error covariance from the new state estimates
	apriori_P_curr = np.zeros((3,3))
	for i in range(0, 2*n):
		apriori_P_curr = apriori_P_curr + np.outer(W_idash[:,i], W_idash[:,i])/(2.*n)

	P_prev = apriori_P_curr # Updating the error covariance after the prediction step
	x_hat_prev = apriori_x_curr # Updating the states after the prediction step
	x_pred_hat[:,t] = x_hat_prev # Storing the prediction state estimate for each time step

	####################################################################################################

	# Update Step

	Z_i = np.empty((3, 2*n))

	for i in range(0, 2*n):
		qy = Quaternions(Y_i[0,i], Y_i[1:,i])
		qy_temp = qy.inverse() * (g * qy) # Applying the measurement model
		Z_i[:,i] = qy_temp.qv

	z_hat_curr = np.mean(Z_i, axis=1) # Computing the mean Z
	v_curr = acc[0:3,t] - z_hat_curr # Innovation obtained from accelerometer data
	ev_meas = (Z_i.T - z_hat_curr.T).T 
	P_xz = np.zeros((3,3))
	P_zz = np.zeros((3,3))
	
	for i in range(0, 2*n):
		P_xz = P_xz + np.outer(W_idash[:,i], ev_meas[:,i])/(2.*n) # Cross correlation matrix
		P_zz = P_zz + np.outer(ev_meas[:,i], ev_meas[:,i])/(2.*n) # Measurement Estimate Covariance
	
	P_vv = P_zz + R # Applying measurement noise covariance to the measurement estimate covariance
	K = np.matmul(P_xz, np.linalg.inv(P_vv)) # Kalman gain

	P_prev = apriori_P_curr - np.matmul(np.matmul(K, P_vv), np.transpose(K)) # Updating the error covariance for the current state estimate
	
	# A posteriori estimate of the state
	x_change = np.matmul(K, v_curr) 
	x_updated = Quaternions(0., x_change/2.).exp() * Quaternions(apriori_x_curr[0], apriori_x_curr[1:])
	
	x_hat_prev = np.append(x_updated.qs, x_updated.qv) # Updating the state estimate after the update step
	q_mean = Quaternions(x_hat_prev[0],x_hat_prev[1:]) # Updating the starting point for quarternion averaging to be the previous state estimate

	x_hat[:,t] = x_hat_prev # Storing the state estimates after the update step

X = np.vstack((ts.T, x_hat, x_pred_hat, x_test_hat))
np.save(filename, X)
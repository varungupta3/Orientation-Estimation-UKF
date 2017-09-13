import numpy as np


class Quaternions:
	
	def __init__(self, qs = 1., qv = np.array([0., 0., 0.])):
		self.qs = qs
		self.qv = qv

	# @classmethod
	# def fromrot(cls, R):
	# 	q = cls.RotToQuat(R):
	# 	return cls(q)

	def norm(self):
		return np.sqrt(self.qs**2 + np.dot(self.qv, self.qv))

	def __add__(self, other):
		q = Quaternions(self.qs + other.qs, self.qv + other.qv)
		norm_q = q.norm()
		return Quaternions(q.qs/norm_q, q.qv/norm_q)

	def __sub__(self, other):
		q = Quaternions(self.qs - other.qs, self.qv - other.qv)
		norm_q = q.norm()
		return Quaternions(q.qs/norm_q, q.qv/norm_q)

	def __mul__(self, other):
		q = Quaternions(self.qs * other.qs - np.dot(self.qv, other.qv), self.qs * other.qv + other.qs * self.qv + np.cross(self.qv, other.qv))
		return q
		# norm_q = q.norm()
		# return Quaternions(q.qs/norm_q, q.qv/norm_q)

	def conjugate(self):
		return Quaternions(self.qs, -self.qv)

	def inverse(self):
		q = self.conjugate()
		norm_q = q.norm()
		if norm_q == 0:
			return Quaternions(np.inf, np.array([np.inf, np.inf, np.inf]))
			# return q
		else:
			return Quaternions(q.qs/(norm_q**2), q.qv/(norm_q**2))

	def exp(self):
		qvec = Quaternions(0, self.qv)
		norm_qv = qvec.norm()
		if norm_qv == 0:
			return Quaternions(np.exp(self.qs), self.qv)
		else:
			q = Quaternions(np.exp(self.qs)*np.cos(norm_qv), np.exp(self.qs) * (self.qv / norm_qv) * np.sin(norm_qv))
			# print q.norm()
			return q
		# norm_q = q.norm()
		# return Quaternions(q.qs/norm_q, q.qv/norm_q)

	def log(self):
		qvec = Quaternions(0, self.qv)
		norm_qv = qvec.norm()
		if norm_qv == 0:
			return Quaternions(np.log(self.qs), self.qv)
		else:
			norm_q = self.norm()
			# print norm_q, norm_qv
			q = Quaternions(np.log(norm_q), self.qv/norm_qv * np.arccos(self.qs/norm_q))
			return q
		# norm_q = q.norm()
		# return Quaternions(q.qs/norm_q, q.qv/norm_q)

	def RotToQuat(self, R):
		self.qs = 1/2.0 * np.sqrt(1 + np.trace(R))
		self.qv = np.array([(R[2,1] - R[1,2])/(4*self.qs), (R[0,2] - R[2,0])/(4*self.qs), (R[1,0] - R[0,1])/(4*self.qs)])
		norm = self.norm()
		self.qs = self.qs/norm
		self.qv = self.qv/norm
		return self
		# return Quaternions(q.qs/norm, q.qv/norm)

	def QuatToRot(self):
		R = np.array([self.qs**2 + self.qv[0]**2 - self.qv[1]**2 - self.qv[2]**2, 2*(self.qv[0]*self.qv[1] - self.qs*self.qv[2]), 2*(self.qs*self.qv[1] + self.qv[0]*self.qv[2])], \
			[2*(self.qv[0]*self.qv[1] + self.qs*self.qv[2]), self.qs**2 - self.qv[0]**2 + self.qv[1]**2 - self.qv[2]**2, 2*(self.qv[1]*self.qv[2] - self.qs*self.qv[0])], \
			[2*(self.qs*self.qv[1] - self.qv[0]*self.qv[2]), 2*(self.qv[1]*self.qv[2] + self.qs*self.qv[0]), self.qs**2 - self.qv[0]**2 - self.qv[1]**2 + self.qv[2]**2])
		return R

	def average(self, others, weights):

		n = np.shape(others)[0]
		# print n
		T = 1000
		qt = Quaternions(self.qs, self.qv) # Average quarternion
		# q0 = Quaternions(self.qs, self.qv) # Initial mean
		# print 'qt = ', qt.qs, qt.qv, qt.norm()
		eps = 0.001
		
		for t in range(0, T):
			# if t != 0:
			# 	print 't = ', t-1
			# 	print 'EV = ', ev
			
			ev = np.zeros((3, n)) # Error Vectors
			# ev0temp = 2*(q0 * qt.inverse()).log().qv
			# ev0norm = np.linalg.norm(ev0temp)
			# if ev0norm == 0:
			# 	ev[:,0] = np.array([0., 0., 0.])
			# else:
			# 	ev[:,0] = (-np.pi + np.mod(ev0norm + np.pi, 2*np.pi))/ev0norm * ev0temp
			# print ev[:,0]

			for i in range(0, n):
				qi = others[i][0] * qt.inverse()
				evtemp = 2 * qi.log().qv
				# print evtemp
				ev_norm = np.linalg.norm(evtemp)
				if ev_norm == 0:
					ev[:,i] = np.array([0., 0., 0.])
				else:
					# if t != 0:
					# 	print 't = ', t, -np.pi + np.mod(ev_norm + np.pi, 2*np.pi), ev_norm
					ev[:,i] = (-np.pi + np.mod(ev_norm + np.pi, 2*np.pi))/ev_norm * evtemp
				# ev(i,:) = ev(i,:) + evtemp * weights[i]
			# print ev
			ev_avg = np.sum(ev * weights, axis = 1) 
			# ev_avg = np.mean(ev, axis=1) # Average along a column

			# if t != 0:
			# 	print 'ev_avg =  ', ev_avg, 'Norm = ', np.linalg.norm(ev_avg) 
			
			qvec = Quaternions(0., ev_avg/2.)
			qt = qvec.exp() * qt
			
			if np.linalg.norm(ev_avg) < eps:
				break

		return qt, ev










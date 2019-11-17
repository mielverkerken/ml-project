from util.constants import *
from libraries.base import *
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from scipy.stats import moment
from scipy.spatial.distance import pdist, cdist, squareform
from functools import wraps
import pdb

def stats(func):
	def wrapper(sample):
		out = []
		for f in func(sample):
			f = f[f == f]
			if len(f) == 0:
				out.extend([np.nan] * 6)
			else:
				diff1, diff2 = (f[(len(f) - 1) // 2] - f[0], f[-1] - f[(len(f) - 1) // 2])
				out.extend([np.max(f), np.min(f), np.mean(f), np.std(f), diff1, diff2])
		return np.array(out)

	return wrapper


def get_arm_angles(pose):
	if np.isnan(pose[l_arm_should][0]) or np.isnan(pose[l_arm_elbow][0]) or np.isnan(pose[l_arm_wrist][0]):
		left_angle = float('NaN')
	else:
		p1 = pose[l_arm_should][:2]
		p2 = pose[l_arm_elbow][:2]
		p3 = pose[l_arm_wrist][:2]

		v0 = np.array(p1) - np.array(p2)
		v1 = np.array(p3) - np.array(p2)

		left_angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

	if np.isnan(pose[r_arm_should][0]) or np.isnan(pose[r_arm_elbow][0]) or np.isnan(pose[r_arm_wrist][0]):
		right_angle = float('NaN')
	else:
		p1 = pose[r_arm_should][:2]
		p2 = pose[r_arm_elbow][:2]
		p3 = pose[r_arm_wrist][:2]

		v0 = np.array(p1) - np.array(p2)
		v1 = np.array(p3) - np.array(p2)

		right_angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
	return np.abs(np.degrees(left_angle)), np.abs(np.degrees(right_angle))


def get_shoulder_angles(pose):
	if np.isnan(pose[neck][0]) or np.isnan(pose[l_arm_should][0]) or np.isnan(pose[l_arm_elbow][0]):
		left_angle = float('NaN')
	else:
		p1 = pose[neck][:2]
		p2 = pose[l_arm_should][:2]
		p3 = pose[l_arm_elbow][:2]

		v0 = np.array(p1) - np.array(p2)
		v1 = np.array(p3) - np.array(p2)

		left_angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
	if np.isnan(pose[neck][0]) or np.isnan(pose[r_arm_should][0]) or np.isnan(pose[r_arm_elbow][0]):
		right_angle = float('NaN')
	else:
		p1 = pose[neck][:2]
		p2 = pose[r_arm_should][:2]
		p3 = pose[r_arm_elbow][:2]

		v0 = np.array(p1) - np.array(p2)
		v1 = np.array(p3) - np.array(p2)

		right_angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
	return np.abs(np.degrees(left_angle)), np.abs(np.degrees(right_angle))


@stats
def get_all_arm_angles(sample):
	arm_angles_left = []
	arm_angles_right = []
	for frame in sample:
		pose, _, _, _ = get_frame_parts(frame)
		arm_angles = get_arm_angles(pose)
		arm_angles_left.append(arm_angles[0])
		arm_angles_right.append(arm_angles[1])
	return np.array(arm_angles_left), np.array(arm_angles_right)


@stats
def get_all_shoulder_angles(sample):
	shoulder_angles_left = []
	shoulder_angles_right = []
	for frame in sample:
		pose, _, _, _ = get_frame_parts(frame)
		shoulder_angles = get_shoulder_angles(pose)
		shoulder_angles_left.append(shoulder_angles[0])
		shoulder_angles_right.append(shoulder_angles[1])
	return np.array(shoulder_angles_left), np.array(shoulder_angles_right)


def get_number_inflections(dy, threshold=0.05):
	number_of_ups_downs = 0
	val_pos = (dy[0] > 0)
	accumulator = 0
	for val in dy:
		if (val_pos != (val > 0)):
			accumulator += abs(val)
			if accumulator > threshold:
				val_pos = (val > 0)
				number_of_ups_downs += 1
				accumulator = 0
	return number_of_ups_downs


def get_hand_movement_raw(sample):
	baricenter_L = []
	baricenter_R = []
	for frame in sample:
		bcl = frame[hand_left_offset:hand_left_offset + hand_left_len, x_index:y_index + 1]
		bcl = bcl[bcl[:, 0] == bcl[:, 0]]
		if len(bcl) == 0:
			bcl = np.array([np.nan] * 2)
		else:
			bcl = bcl.mean(axis=0)
		baricenter_L.append(bcl)

		bcr = frame[hand_right_offset:hand_right_offset + hand_right_len, x_index:y_index + 1]
		bcr = bcr[bcr[:, 0] == bcr[:, 0]]
		if len(bcr) == 0:
			bcr = np.array([np.nan] * 2)
		else:
			bcr = bcr.mean(axis=0)
		baricenter_R.append(bcr)

	dx_L, dy_L = np.diff(np.array(baricenter_L).T)
	dx_R, dy_R = np.diff(np.array(baricenter_R).T)

	return (dx_L, dx_R, dy_L, dy_R)


def get_hand_movement_raw_uncertain(sample):
	baricenter_L = []
	baricenter_R = []
	for frame in sample:
		bcl = frame[hand_left_offset:hand_left_offset + hand_left_len, :]
		bcl_uncertain = bcl[:, 2:][bcl[:, 0] == bcl[:, 0]]
		bcl = bcl[:, 0:2]
		bcl = bcl[bcl[:, 0] == bcl[:, 0]]

		if len(bcl) == 0:
			bcl = np.array([np.nan] * 2)
		else:
			bcl = (bcl * bcl_uncertain).sum(axis=0) / bcl_uncertain.sum(axis=0)
		baricenter_L.append(bcl)

		bcr = frame[hand_right_offset:hand_right_offset + hand_right_len, :]
		bcr_uncertain = bcr[:, 2:][bcr[:, 0] == bcr[:, 0]]
		bcr = bcr[:, 0:2]
		bcr = bcr[bcr[:, 0] == bcr[:, 0]]

		if len(bcr) == 0:
			bcr = np.array([np.nan] * 2)
		else:
			bcr = (bcr * bcr_uncertain).sum(axis=0) / bcr_uncertain.sum(axis=0)
		baricenter_R.append(bcr)

	dx_L, dy_L = np.diff(np.array(baricenter_L).T)
	dx_R, dy_R = np.diff(np.array(baricenter_R).T)

	return (dx_L, dx_R, dy_L, dy_R)


@stats
def get_hand_movement(sample):
	return get_hand_movement_raw(sample)


@stats
def get_hand_movement_uncertain(sample):
	return get_hand_movement_raw_uncertain(sample)


def arclength(x):
	Sum = 0
	x = x[x[:, 0] == x[:, 0]]
	size = x.shape[0]
	if size == 0:
		return float('nan')
	for i in range(1, size):
		Sum += dist(x[i], x[i - 1])
	return dist(x[-1], x[0]) / (Sum + 1e-5)


@stats
def finger_openness(sample):
	# @title finger open/closed
	n_fingers = 10
	finger_openness_feature = np.zeros((n_fingers, len(sample)))
	for j, frame in enumerate(sample):
		_, _, hand_L, hand_R = get_frame_parts(frame)
		for k in range(int(n_fingers / 2)):
			# Left Hand
			finger_openness_feature[k, j] = arclength(hand_L[1 + 4 * k:5 + 4 * k])
			# Right Hand
			finger_openness_feature[k + 5, j] = arclength(hand_R[1 + 4 * k:5 + 4 * k])
	return finger_openness_feature


@stats
def shoulder_wrist_y(sample):
	R = []
	L = []
	for frame in sample:
		body, _, _, _ = get_frame_parts(frame)
		c_shoulder = body[[1, 2, 5], 1]
		c_shoulder = c_shoulder[c_shoulder == c_shoulder]
		if len(c_shoulder) == 0:
			c_shoulder = float('nan')
		else:
			c_shoulder = c_shoulder.mean()
		r_wrist = body[4, 1]
		l_wrist = body[7, 1]
		d_r = c_shoulder - r_wrist
		d_l = c_shoulder - l_wrist
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def shoulder_wrist_y_uncertain(sample):
	R = []
	L = []
	for frame in sample:
		body, _, _, _ = get_frame_parts(frame)
		c_shoulder = body[[1, 2, 5], 1]
		c_shoulder_uncertain = body[[1, 2, 5], 2][c_shoulder == c_shoulder]
		c_shoulder = c_shoulder[c_shoulder == c_shoulder]
		if len(c_shoulder) == 0:
			c_shoulder = float('nan')
		else:
			c_shoulder = (c_shoulder_uncertain * c_shoulder).sum(axis=0) / c_shoulder_uncertain.sum(axis=0)
		r_wrist = body[4, 2] * body[4, 1]
		l_wrist = body[7, 2] * body[7, 1]
		d_r = c_shoulder - r_wrist
		d_l = c_shoulder - l_wrist
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def head_hand(sample):
	R = []
	L = []
	head = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		f_head = f_head[f_head[:, 0] == f_head[:, 0]][:, 0:2]
		if len(f_head):
			f_head = f_head.mean(axis=0)
			head[i] = f_head

	head = head[head[:, 0] == head[:, 0]]
	if len(head):
		head = np.mean(head)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]][:, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]][:, 0:2]
		if len(r_hand) == 0:
			d_r = np.nan
		else:
			r_hand = r_hand.mean(axis=0)
			d_r = dist(head, r_hand)

		if len(l_hand) == 0:
			d_l = np.nan
		else:
			l_hand = l_hand.mean(axis=0)
			d_l = dist(head, l_hand)

		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def head_hand_uncertain(sample):
	R = []
	L = []
	head = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		f_head_uncertain = f_head[:, 2:][f_head[:, 0] == f_head[:, 0]]
		f_head = f_head[:, 0:2]
		f_head = f_head[f_head[:, 0] == f_head[:, 0]]

		if len(f_head):
			f_head = (f_head_uncertain * f_head).sum(axis=0) / f_head_uncertain.sum(axis=0)
			head[i] = f_head

	head = head[head[:, 0] == head[:, 0]]
	if len(head):
		head = np.mean(head)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand_uncertain = r_hand[:, 2:][r_hand[:, 0] == r_hand[:, 0]]
		r_hand = r_hand[:, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]

		l_hand_uncertain = l_hand[:, 2:][l_hand[:, 0] == l_hand[:, 0]]
		l_hand = l_hand[:, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]

		if len(r_hand) == 0:
			d_r = np.nan
		else:
			r_hand = (r_hand_uncertain * r_hand).sum(axis=0) / r_hand_uncertain.mean(axis=0)
			d_r = dist(head, r_hand)

		if len(l_hand) == 0:
			d_l = np.nan
		else:
			(l_hand_uncertain * l_hand).sum(axis=0) / l_hand_uncertain.mean(axis=0)
			d_l = dist(head, l_hand)

		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def var_hands(sample):
	Rx = []
	Ry = []
	Lx = []
	Ly = []
	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]][:, 0:2]
		if len(r_hand):
			r_hand = r_hand.var(axis=0)
		else:
			r_hand = [np.nan] * 2
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]][:, 0:2]
		if len(l_hand):
			l_hand = l_hand.var(axis=0)
		else:
			l_hand = [np.nan] * 2
		Rx.append(r_hand[0])
		Ry.append(r_hand[1])
		Lx.append(l_hand[0])
		Ly.append(l_hand[1])
	return [np.array(Lx), np.array(Ly), np.array(Rx), np.array(Ry)]


@stats
def chin_thumb(sample):
	R = []
	L = []
	chin = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		f_head = f_head[7:10, 0:2]
		f_head = f_head[f_head[:, 0] == f_head[:, 0]]
		if len(f_head):
			f_head = f_head.mean(axis=0)
			chin[i] = f_head

	chin = chin[chin[:, 0] == chin[:, 0]]
	if len(chin):
		chin = np.mean(chin)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[2:5, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
		if len(r_hand):
			r_hand = r_hand.mean(axis=0)
		else:
			r_hand = np.nan
		l_hand = l_hand[2:5, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
		if len(l_hand):
			l_hand = l_hand.mean(axis=0)
		else:
			l_hand = np.nan
		d_r = dist(chin, r_hand)
		d_l = dist(chin, l_hand)
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def chin_thumb_uncertain(sample):
	R = []
	L = []
	chin = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)

		f_head = f_head[7:10, :]
		f_head_uncertain = f_head[:, 2:][f_head[:, 0] == f_head[:, 0]]
		f_head = f_head[:, 0:2]
		f_head = f_head[f_head[:, 0] == f_head[:, 0]]

		if len(f_head):
			f_head = (f_head_uncertain * f_head).sum(axis=0) / f_head_uncertain.sum(axis=0)
			chin[i] = f_head

	chin = chin[chin[:, 0] == chin[:, 0]]
	if len(chin):
		chin = np.mean(chin)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)

		r_hand = r_hand[2:5, :]
		r_hand_uncertain = r_hand[:, 2:][r_hand[:, 0] == r_hand[:, 0]]
		r_hand = r_hand[:, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]

		if len(r_hand):
			r_hand = (r_hand_uncertain * r_hand).sum(axis=0) / r_hand_uncertain.mean(axis=0)
		else:
			r_hand = np.nan

		l_hand = l_hand[2:5, :]
		l_hand_uncertain = l_hand[:, 2:][l_hand[:, 0] == l_hand[:, 0]]
		l_hand = l_hand[:, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]

		if len(l_hand):
			l_hand = (l_hand_uncertain * l_hand).sum(axis=0) / l_hand_uncertain.mean(axis=0)
		else:
			l_hand = np.nan
		d_r = dist(chin, r_hand)
		d_l = dist(chin, l_hand)
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def mouth_distance(sample):
	distances = []
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		f_head = f_head[:, 0:2]
		# f_head = f_head[f_head[:, 0] == f_head[:, 0]]
		keypoint_pairs = [[50, 58], [51, 57], [52, 56], [61, 67], [62, 66], [63, 65]]
		d = []
		for a, b in keypoint_pairs:
			if not np.isnan(f_head[a, :]).any() and not np.isnan(f_head[b, :]).any():
				d.append(dist(f_head[a, :], f_head[b, :]))
		if len(d) > 0:
			distances.append(np.mean(d))
	return [np.array(distances)]


@stats
def mouth_distance_uncertain(sample):
	distances = []
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		keypoint_pairs = [[50, 58], [51, 57], [52, 56], [61, 67], [62, 66], [63, 65]]
		d = []
		for a, b in keypoint_pairs:
			if not np.isnan(f_head[a, :]).any() and not np.isnan(f_head[b, :]).any():
				d.append(np.sqrt(f_head[a, 2] * f_head[b, 2]) * dist(f_head[a, :2], f_head[b, :2]))
		if len(d) > 0:
			distances.append(np.mean(d))
	return [np.array(distances)]


@stats
def mouth_index(sample):
	R = []
	L = []
	mouth = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		_, f_head, _, _ = get_frame_parts(frame)
		f_head = f_head[48:68, 0:2]
		f_head = f_head[f_head[:, 0] == f_head[:, 0]]
		if len(f_head):
			f_head = f_head.mean(axis=0)
			mouth[i] = f_head

	mouth = mouth[mouth[:, 0] == mouth[:, 0]]
	if len(mouth):
		mouth = np.mean(mouth)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[6:9, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
		if len(r_hand):
			r_hand = r_hand.mean(axis=0)
		else:
			r_hand = np.nan

		l_hand = l_hand[6:9, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
		if len(l_hand):
			l_hand = l_hand.mean(axis=0)
		else:
			l_hand = np.nan

		d_r = dist(mouth, r_hand)
		d_l = dist(mouth, l_hand)
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)

@stats
def mouth_index_uncertain(sample):
	R = []
	L = []
	mouth = np.zeros((len(sample), 2))
	for i, frame in enumerate(sample):
		# pdb.set_trace()
		_, f_head, _, _ = get_frame_parts(frame)
		f_head = f_head[48:68, :]
		f_head_uncertain = f_head[:, 2:][f_head[:, 0] == f_head[:, 0]]
		f_head = f_head[:, 0:2]
		f_head = f_head[f_head[:, 0] == f_head[:, 0]]

		if len(f_head):
			f_head = (f_head_uncertain * f_head).sum(axis=0) / f_head_uncertain.sum(axis=0)
			mouth[i] = f_head

	mouth = mouth[mouth[:, 0] == mouth[:, 0]]
	if len(mouth):
		mouth = np.mean(mouth)
	else:
		return np.array([[float('nan')], [float('nan')]])

	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[6:9, :]
		r_hand_uncertain = r_hand[:, 2:][r_hand[:, 0] == r_hand[:, 0]]
		r_hand = r_hand[:, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
		if len(r_hand):
			r_hand = (r_hand_uncertain * r_hand).sum(axis=0) / r_hand_uncertain.mean(axis=0)
		else:
			r_hand = np.nan

		l_hand = l_hand[6:9, :]
		l_hand_uncertain = l_hand[:, 2:][l_hand[:, 0] == l_hand[:, 0]]
		l_hand = l_hand[:, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
		if len(l_hand):
			l_hand = (l_hand_uncertain * l_hand).sum(axis=0) / l_hand_uncertain.mean(axis=0)
		else:
			l_hand = np.nan

		d_r = dist(mouth, r_hand)
		d_l = dist(mouth, l_hand)
		R.append(d_r)
		L.append(d_l)

	return np.array(L), np.array(R)


@stats
def thumb_pink(sample):
	R = []
	L = []
	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)

		r_hand1 = r_hand[2:5, 0:2]
		r_hand1 = r_hand1[r_hand1[:, 0] == r_hand1[:, 0]]

		if len(r_hand1):
			r_hand1 = r_hand1.mean(axis=0)
		else:
			r_hand1 = np.nan
		l_hand1 = l_hand[2:5, 0:2]
		l_hand1 = l_hand1[l_hand1[:, 0] == l_hand1[:, 0]]
		if len(l_hand1):
			l_hand1 = l_hand1.mean(axis=0)
		else:
			l_hand1 = np.nan

		r_hand2 = r_hand[18:21, 0:2]
		r_hand2 = r_hand2[r_hand2[:, 0] == r_hand2[:, 0]]
		if len(r_hand2):
			r_hand2 = r_hand2.mean(axis=0)
		else:
			r_hand2 = np.nan

		l_hand2 = l_hand[18:21, 0:2]
		l_hand2 = l_hand2[l_hand2[:, 0] == l_hand2[:, 0]]
		if len(l_hand2):
			l_hand2 = l_hand2.mean(axis=0)
		else:
			l_hand2 = np.nan

		d_r = dist(r_hand1, r_hand2)
		d_l = dist(l_hand1, l_hand2)
		R.append(d_r)
		L.append(d_l)
	return np.array(L), np.array(R)


@stats
def thumb_pink_uncertain(sample):
	R = []
	L = []
	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)

		r_hand1 = r_hand[2:5, :]
		r_hand1_uncertain = r_hand1[:, 2:][r_hand1[:, 0] == r_hand1[:, 0]]
		r_hand1 = r_hand1[:, 0:2]
		r_hand1 = r_hand1[r_hand1[:, 0] == r_hand1[:, 0]]

		if len(r_hand1):
			r_hand1 = (r_hand1_uncertain * r_hand1).sum(axis=0) / r_hand1_uncertain.sum(axis=0)
		else:
			r_hand1 = np.nan

		l_hand1 = l_hand[2:5, :]
		l_hand1_uncertain = l_hand1[:, 2:][l_hand1[:, 0] == l_hand1[:, 0]]
		l_hand1 = l_hand1[:, 0:2]
		l_hand1 = l_hand1[l_hand1[:, 0] == l_hand1[:, 0]]

		if len(l_hand1):
			l_hand1 = (l_hand1_uncertain * l_hand1).sum(axis=0) / l_hand1_uncertain.sum(axis=0)
		else:
			l_hand1 = np.nan

		r_hand2 = r_hand[18:21, :]
		r_hand2_uncertain = r_hand2[:, 2:][r_hand2[:, 0] == r_hand2[:, 0]]
		r_hand2 = r_hand2[:, 0:2]
		r_hand2 = r_hand2[r_hand2[:, 0] == r_hand2[:, 0]]

		if len(r_hand2):
			r_hand2 = (r_hand2 * r_hand2_uncertain).sum(axis=0) / r_hand2_uncertain.sum(axis=0)
		else:
			r_hand2 = np.nan

		l_hand2 = l_hand[18:21, :]
		l_hand2_uncertain = l_hand2[:, 2:][l_hand2[:, 0] == l_hand2[:, 0]]
		l_hand2 = l_hand2[:, 0:2]
		l_hand2 = l_hand2[l_hand2[:, 0] == l_hand2[:, 0]]

		if len(l_hand2):
			l_hand2 = (l_hand2_uncertain * l_hand2).sum(axis=0) / l_hand2_uncertain.sum(axis=0)
		else:
			l_hand2 = np.nan

		d_r = dist(r_hand1, r_hand2)
		d_l = dist(l_hand1, l_hand2)
		R.append(d_r)
		L.append(d_l)
	return np.array(L), np.array(R)


@stats
def index_index(sample):
	out = []
	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)
		r_hand = r_hand[6:9, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
		if len(r_hand):
			r_hand = r_hand.mean(axis=0)
		else:
			r_hand = np.nan
		l_hand = l_hand[6:9, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
		if len(l_hand):
			l_hand = l_hand.mean(axis=0)
		else:
			l_hand = np.nan
		d = dist(r_hand, l_hand)
		out.append(d)
	return [np.array(out)]


@stats
def index_index_uncertain(sample):
	out = []
	for frame in sample:
		_, _, r_hand, l_hand = get_frame_parts(frame)

		r_hand = r_hand[6:9, :]
		r_hand_uncertain = r_hand[:, 2:][r_hand[:, 0] == r_hand[:, 0]]
		r_hand = r_hand[:, 0:2]
		r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]

		if len(r_hand):
			r_hand = (r_hand_uncertain * r_hand).sum(axis=0) / r_hand_uncertain.sum(axis=0)
		else:
			r_hand = np.nan

		l_hand = l_hand[6:9, :]
		l_hand_uncertain = l_hand[:, 2:][l_hand[:, 0] == l_hand[:, 0]]
		l_hand = l_hand[:, 0:2]
		l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]

		if len(l_hand):
			l_hand = (l_hand_uncertain * l_hand).sum(axis=0) / l_hand_uncertain.sum(axis=0)
		else:
			l_hand = np.nan
		d = dist(r_hand, l_hand)
		out.append(d)
	return [np.array(out)]


@stats
def wrist_wrist_x(sample):
	out = []
	for frame in sample:
		body, _, _, _ = get_frame_parts(frame)
		r_wrist = body[4, 0]
		l_wrist = body[7, 0]
		d = l_wrist - r_wrist
		out.append(d)

	return [np.array(out)]


@stats
def wrist_wrist_x_uncertain(sample):
	out = []
	for frame in sample:
		body, _, _, _ = get_frame_parts(frame)
		r_wrist = body[4, 2] * body[4, 0]
		l_wrist = body[7, 0] * body[7, 0]
		d = l_wrist - r_wrist
		out.append(d)

	return [np.array(out)]


def confidence_hands(sample):
	# Returns mean confidence of x and y coordinate over all frames of a sample. First value is for left hand, second for right hand.
	conf_left = np.nanmean(sample[:, np.arange(hand_left_offset, hand_left_offset + hand_left_len), c_index])
	conf_right = np.nanmean(sample[:, np.arange(hand_right_offset, hand_right_offset + hand_right_len), c_index])
	return [conf_left, conf_right]


def number_of_frames(sample):
	return [len(sample)]


@stats
def reverse_hand_movement(sample):
	(dx_L, dx_R, dy_L, dy_R) = get_hand_movement_raw(sample)
	X = dx_L * dx_R
	Y = dy_L * dy_R
	return (X, Y)


def generate_feature_matrix(all_samples):
	NUM_SAMPLES = len(all_samples)

	# regular statistical features                         #singular features
	COLUMNS = (NUM_FEATURES - NUM_FEATURES_WITHOUT_STATS) * NUM_STATS + NUM_FEATURES_WITHOUT_STATS

	FEATURE_MATRIX = np.zeros((NUM_SAMPLES, COLUMNS))

	for i, sample in enumerate(tqdm(all_samples)):
		sample_row = []
		# expect 1 feature for num frames
		sample_row.extend(number_of_frames(sample))

		# expect 4 features for the inflection points
		# only worth calculating when len(dy|dx) > 1
		if (len(sample) > 2):
			(dx_L, dx_R, dy_L, dy_R) = get_hand_movement_raw(sample)
			sample_row.extend([get_number_inflections(dy_L), get_number_inflections(dy_R), get_number_inflections(dx_L), get_number_inflections(dx_R)])
		else:
			sample_row.extend([0] * 4)

		# expect 2 features for hand confidence
		sample_row.extend(confidence_hands(sample))

		# expect 12 features for arm angles
		sample_row.extend(get_all_arm_angles(sample))
		# expect 12 features for shoulder angles
		sample_row.extend(get_all_shoulder_angles(sample))

		# expect 24 features for the hand movement
		if (len(sample) > 1):
			sample_row.extend(get_hand_movement(sample))
		else:
			sample_row.extend([0] * 24)

		# expect 60 features for finger openness
		sample_row.extend(finger_openness(sample))

		# expect 12 features for shoulder wrist
		sample_row.extend(shoulder_wrist_y(sample))

		# expect 12 featurs for head hand
		sample_row.extend(head_hand(sample))

		# expect 24 featurs for hands variation
		sample_row.extend(var_hands(sample))

		# expect 12 features for chin thumb
		sample_row.extend(chin_thumb(sample))

		# expect 12 features for mouth index
		sample_row.extend(mouth_index(sample))

		# expect 12 features for thumb pink
		sample_row.extend(thumb_pink(sample))

		# expect 6 features for index index
		sample_row.extend(index_index(sample))

		# expect 6 features for index index
		sample_row.extend(wrist_wrist_x(sample))

		# expect 12 featurs for reverse hand movement
		if (len(sample) > 1):
			sample_row.extend(reverse_hand_movement(sample))
		else:
			sample_row.extend([0] * 12)

		# expect 6 features for distance between upper and lower lips
		sample_row.extend(mouth_distance(sample))

		# transform to numpy array
		FEATURE_MATRIX[i] = np.array(sample_row)
	return FEATURE_MATRIX


def generate_feature_matrix_uncertain(all_samples):
	NUM_SAMPLES = len(all_samples)

	# regular statistical features                         #singular features
	COLUMNS = (NUM_FEATURES - NUM_FEATURES_WITHOUT_STATS) * NUM_STATS + NUM_FEATURES_WITHOUT_STATS
	COLUMNS = 24 + 12 + 12 + 12 + 12 + 12 + 6 + 6 + 6
	# COLUMNS = (NUM_FEATURES - NUM_FEATURES_WITHOUT_STATS) * NUM_STATS + NUM_FEATURES_WITHOUT_STATS + 6 + 12

	FEATURE_MATRIX = np.zeros((NUM_SAMPLES, COLUMNS))

	for i, sample in enumerate(tqdm(all_samples)):
		sample_row = []

		# expect 24 features for the hand movement
		if (len(sample) > 1):
			sample_row.extend(get_hand_movement_uncertain(sample))  # Works
		else:
			sample_row.extend([0] * 24)

		# expect 12 features for shoulder wrist
		sample_row.extend(shoulder_wrist_y_uncertain(sample))  # Works

		# expect 12 features for head hand
		sample_row.extend(head_hand_uncertain(sample))  # Works

		# expect 12 features for chin thumb
		sample_row.extend(chin_thumb_uncertain(sample))  # Works

		# expect 12 features for mouth index
		sample_row.extend(mouth_index_uncertain(sample))  # Works

		# expect 12 features for thumb pink
		sample_row.extend(thumb_pink_uncertain(sample))  # Works

		# expect 6 features for index index
		sample_row.extend(index_index_uncertain(sample))  # Works

		# expect 6 features for index index
		sample_row.extend(wrist_wrist_x_uncertain(sample))  # Works

		# expect 6 features for distance between upper and lower lips
		sample_row.extend(mouth_distance_uncertain(sample))  # Works

		# transform to numpy array
		FEATURE_MATRIX[i] = np.array(sample_row)
	return FEATURE_MATRIX


def extract_keypoint_means(samples_list):
	labels = []
	pose_means = [np.nanmean(sample[:, pose_offset:pose_offset + 9, :], axis=0) for sample in samples_list]
	pose_means = np.stack(pose_means, axis=0).reshape((len(samples_list), -1))
	labels.extend(["keypoint_" + str(i) + "_" + j for i in range(pose_offset, pose_offset + 9) for j in ["x", "y", "c"]])

	head_means = [np.nanmean(sample[:, face_offset:face_offset + face_len, :], axis=0) for sample in samples_list]
	head_means = np.stack(head_means, axis=0).reshape((len(samples_list), -1))
	labels.extend(["keypoint_" + str(i) + "_" + j for i in range(face_offset, face_offset + face_len) for j in ["x", "y", "c"]])

	left_means = [np.nanmean(sample[:, hand_left_offset:hand_left_offset + hand_left_len, :], axis=0) for sample in samples_list]
	left_means = np.stack(left_means, axis=0).reshape((len(samples_list), -1))
	labels.extend(["keypoint_" + str(i) + "_" + j for i in range(hand_left_offset, hand_left_offset + hand_left_len) for j in ["x", "y", "c"]])

	right_means = [np.nanmean(sample[:, hand_right_offset:hand_right_offset + hand_right_len, :], axis=0) for sample in samples_list]
	right_means = np.stack(right_means, axis=0).reshape((len(samples_list), -1))
	labels.extend(["keypoint_" + str(i) + "_" + j for i in range(hand_right_offset, hand_right_offset + hand_right_len) for j in ["x", "y", "c"]])
	features = np.concatenate((pose_means, head_means, left_means, right_means), axis=1)
	return features, labels


def transform_to_panda_dataframe(MATRIX):
	df = pd.DataFrame()
	for feature_index, feature_col in enumerate(MATRIX.T):
		if feature_index < NUM_FEATURES_WITHOUT_STATS:
			df[FEATURE_LIST[feature_index]] = feature_col
		else:
			actual_feature_index = NUM_FEATURES_WITHOUT_STATS + ((feature_index - NUM_FEATURES_WITHOUT_STATS) // NUM_STATS)
			stat_index = (feature_index - NUM_FEATURES_WITHOUT_STATS) % NUM_STATS
			column_name = FEATURE_LIST[actual_feature_index] + '_' + STAT_LIST[stat_index]
			df[column_name] = feature_col
	return df


def concat_keypoint_means(dataframe, all_samples):
	keypoint_means, keypoint_labels = extract_keypoint_means(all_samples)
	total_features = keypoint_means.shape[1]
	df = pd.DataFrame(data=keypoint_means, columns=keypoint_labels)
	X_new = pd.concat([dataframe, df], axis=1)
	return X_new


def concat_keypoint_means_numpy(X, all_samples):
	keypoint_means, keypoint_labels = extract_keypoint_means(all_samples)
	total_features = keypoint_means.shape[1]
	df = pd.DataFrame(data=keypoint_means, columns=keypoint_labels)
	X_new = np.concatenate((keypoint_means, X), axis=1)
	return X_new


# Test Arthur
def moments(func):
	@wraps(func)
	def wrapper(sample, indices_relevant):
		return np.stack([func(sample, indices_relevant).sum(0)] + [moment(func(sample, indices_relevant), axis=0, moment=i) for i in np.arange(2, 5)])
	return wrapper


@moments
def moments_frame(sample, indices_relevant):
	shape_ =  sample[:, indices_relevant, :2].shape
	return sample[:, indices_relevant, :2].reshape(shape_[0], shape_[1] * shape_[2])
	#np.stack([sample[:, indices_relevant, :2].mean(axis=0)] + [moment(sample[:, indices_relevant, :2], axis=0, moment=i) for i in np.arange(2, 5)])


def average_frame(sample, indices_relevant):
	return sample[:, indices_relevant, :2].mean(axis=0)


@moments
def get_dist_mean(sample, indices_relevant, body_relevant=[0, 1, 2, 3]):
	# Calculate mean of boy part for each frame. Weighted by the their confidence
	body_part_count = 4
	start_ind = [0, 25, 95, 116, 137]  # pose, face, left-hand, right-hand
	matrix = np.zeros((len(sample), body_part_count, 2))  # 2 (x,y)
	# Compute mean for each body-part
	for b in range(body_part_count):
		# If all coordinates are uncertain for the body part frame is worthless
		mask = (sample[:, start_ind[b]:start_ind[b + 1], 2].sum(axis=1) > 0)
		if mask.sum() != 0:  # Complete sample has no frames with corresponding body part
			# matrix = np.zeros((mask.sum(), body_part_count, 2))  # 2 = (x, y)
			matrix[mask, b, :] = (sample[mask, start_ind[b]:start_ind[b + 1], :2]).sum(axis=1) / len(mask)
			# matrix = np.delete(matrix, ~mask, 0)  # remove all other frames
	dist_mean = np.stack([pdist(frame_mean[body_relevant]) for frame_mean in matrix])
	##dist_mean_features = np.stack([dist_mean.sum(0)] + [moment(dist_mean, axis=0, moment=i) for i in np.arange(2, 5)])
	# dist_mat.sum(0), could be dist_mat.mean(0) but not using the mean works better. Implicit use of the number of
	# frames, time of a sample
	return dist_mean


@moments
def get_dist_mean_uncertain(sample, indices_relevant, body_relevant=[0, 1, 2, 3]):
	# Calculate mean of boy part for each frame. Weighted by the their confidence
	body_part_count = 4
	start_ind = [0, 25, 95, 116, 137]  # pose, face, left-hand, right-hand
	matrix = np.zeros((len(sample), body_part_count, 2))  # 2 (x,y)
	# Compute mean for each body-part
	for b in range(body_part_count):
		# If all coordinates are uncertain for the body part frame is worthless
		mask = (sample[:, start_ind[b]:start_ind[b + 1], 2].sum(axis=1) > 0)
		if mask.sum() != 0:  # Complete sample has no frames with corresponding body part
			# matrix = np.zeros((mask.sum(), body_part_count, 2))  # 2 = (x, y)
			matrix[mask, b, :] = (sample[mask, start_ind[b]:start_ind[b + 1], 2:] * sample[mask,
																					start_ind[b]:start_ind[b + 1], :2]).sum(
				axis=1) / sample[mask, start_ind[b]:start_ind[b + 1], 2:].sum(axis=1)
			# matrix = np.delete(matrix, ~mask, 0)  # remove all other frames
	dist_mean = np.stack([pdist(frame_mean[body_relevant]) for frame_mean in matrix])
	##dist_mean_features = np.stack([dist_mean.sum(0)] + [moment(dist_mean, axis=0, moment=i) for i in np.arange(2, 5)])
	# dist_mat.sum(0), could be dist_mat.mean(0) but not using the mean works better. Implicit use of the number of
	# frames, time of a sample
	return dist_mean


@moments
def get_dist_mat_features(sample, indices_relevant):  # Confidence is included by u[2] * v[2]
	dist_mat = np.stack([pdist(frame[indices_relevant][:, :2]) for frame in sample])
	##dist_mat_features = np.stack([dist_mat.sum(0)] + [moment(dist_mat, axis=0, moment=i) for i in np.arange(2, 5)])
	return dist_mat


@moments
def get_dist_mat_features_uncertain(sample, indices_relevant):  # Confidence is included by u[2] * v[2]
	dist_mat = np.stack([squareform(
		np.sqrt(np.abs(frame[indices_relevant][:, 2][:, None] * frame[indices_relevant][:, 2][None, :])), checks=False) * pdist(
		frame[indices_relevant][:, :2]) for frame in sample])
	##dist_mat_features = np.stack([dist_mat.sum(0)] + [moment(dist_mat, axis=0, moment=i) for i in np.arange(2, 5)])
	return dist_mat


@moments
def get_time_evolution_features(sample, indices_relevant):  # Confidence is included by sample[1:][:, :, 2:]) * sample[:-1][:, :, 2:]
	dist_mat = np.zeros((len(sample), len(indices_relevant)))
	if len(sample) != 1:
		dist_mat = np.sqrt(((sample[1:][:, indices_relevant, :2] - sample[:-1][:, indices_relevant, :2]) ** 2).sum(-1))
	## dist_mat_features = np.stack([dist_mat.sum(0)] + [moment(dist_mat, axis=0, moment=i) for i in np.arange(2, 5)])
	return dist_mat


@moments
def get_time_evolution_features_uncertain(sample, indices_relevant):  # Confidence is included by sample[1:][:, :, 2:]) * sample[:-1][:, :, 2:]
	dist_mat = np.zeros((len(sample), len(indices_relevant)))
	if len(sample) != 1:
		dist_mat = np.sqrt((np.abs(sample[1:][:, indices_relevant, 2:] * sample[:-1][:, indices_relevant, 2:]) * (
				sample[1:][:, indices_relevant, :2] - sample[:-1][:, indices_relevant, :2]) ** 2).sum(-1))
	## dist_mat_features = np.stack([dist_mat.sum(0)] + [moment(dist_mat, axis=0, moment=i) for i in np.arange(2, 5)])
	return dist_mat


@moments
def get_time_evolution_directions(sample, indices_relevant):  # Similar as above but now for x and y distance
	x_dist = np.zeros((len(sample), len(indices_relevant)))  # difference in x
	y_dist = np.zeros((len(sample), len(indices_relevant)))  # difference in y
	if len(sample) != 1:
		x_dist = np.sqrt((sample[1:][:, indices_relevant, 0] - sample[:-1][:, indices_relevant, 0]) ** 2)
		y_dist = np.sqrt((sample[1:][:, indices_relevant, 1] - sample[:-1][:, indices_relevant, 1]) ** 2)
	xy_features = np.hstack((x_dist, y_dist))
	#x_features = np.stack([x_dist.sum(0)] + [moment(x_dist, axis=0, moment=i) for i in np.arange(2, 5)])
	#y_features = np.stack([y_dist.sum(0)] + [moment(y_dist, axis=0, moment=i) for i in np.arange(2, 5)])
	return xy_features
	#np.concatenate([x_features, y_features], axis=0)


@moments
def get_time_evolution_directions_uncertain(sample, indices_relevant):  # Similar as above but now for x and y distance
	x_dist = np.zeros((len(sample), len(indices_relevant)))  # difference in x
	y_dist = np.zeros((len(sample), len(indices_relevant)))  # difference in y
	if len(sample) != 1:
		x_dist = np.sqrt(np.abs(sample[1:][:, indices_relevant, 2] * sample[:-1][:, indices_relevant, 2]) * (
				sample[1:][:, indices_relevant, 0] - sample[:-1][:, indices_relevant, 0]) ** 2)
		y_dist = np.sqrt(np.abs(sample[1:][:, indices_relevant, 2] * sample[:-1][:, indices_relevant, 2]) * (
				sample[1:][:, indices_relevant, 1] - sample[:-1][:, indices_relevant, 1]) ** 2)
	xy_features = np.hstack((x_dist, y_dist))
	#x_features = np.stack([x_dist.sum(0)] + [moment(x_dist, axis=0, moment=i) for i in np.arange(2, 5)])
	#y_features = np.stack([y_dist.sum(0)] + [moment(y_dist, axis=0, moment=i) for i in np.arange(2, 5)])
	return xy_features
	#np.concatenate([x_features, y_features], axis=0)


def generate_feature_matrix_dist(all_samples, feature_func):
	# Constants
	indices_pose = np.arange(0, 25)
	indices_face = np.arange(25, 95)
	indices_lh = np.arange(95, 116)
	indices_rh = np.arange(116, 137)

	# Considered keypoints, can be experimented with
	indices_relevant = np.hstack([indices_lh, indices_rh])

	# Number of features
	train_features = []
	for ind, sample in enumerate(tqdm(all_samples)):
		train_features.append(feature_func(sample, indices_relevant).flatten())
	num_features = len(train_features[0])
	FEATURE_MATRIX = np.vstack(train_features)
	print(f"Dimensions features ({feature_func.__name__}): \n {FEATURE_MATRIX.shape}")
	# print(f"Nan check: {np.isnan(FEATURE_MATRIX).sum()}")
	# = np.hstack(train_features) #[:]
	return FEATURE_MATRIX, num_features


def transform_to_panda_dataframe_dist(MATRIX, num_features, feature_func):
	df = pd.DataFrame()
	for i in range(num_features):
		column_name = f"{feature_func.__name__} {i}"
		df[column_name] = MATRIX.T[i]
	return df



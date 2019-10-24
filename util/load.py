import numpy as np
import csv
from os.path import join as pjoin
from glob import glob

# PATHS
DATA_DIR = './data'
POSE_DIR = './data/pose'

# LOAD DATASET AND LABELS
def load_data():
	dataset_file = pjoin(DATA_DIR, 'labels.csv')

	all_samples = []
	all_labels = []
	all_persons = []
	label_to_gloss = {}

	with open(dataset_file) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		for row in reader:
			name, _gloss, label, _person = row
			sample = np.load(pjoin(POSE_DIR, 'train', name+'.npy'))
			all_samples.append(sample)
			all_labels.append(label)
			all_persons.append(_person)
		
			if int(label) not in label_to_gloss.keys():
				label_to_gloss[int(label)] = _gloss
                
	all_labels = np.array(all_labels).astype(int)
	all_samples = np.array(all_samples)
	all_persons = np.array(all_persons)


	all_test_files = sorted(glob(pjoin(POSE_DIR, 'test', '*.npy')))  

	test_samples = []
	for numpy_file in all_test_files:
		sample = np.load(numpy_file)
		test_samples.append(sample)

	print("%d samples and labels loaded" %(all_labels.shape[0]))
	return all_samples, all_labels, all_persons, label_to_gloss, test_samples

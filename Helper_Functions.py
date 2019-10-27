import numpy as np
import random 
import itertools 
import os
import sys
import pickle
import matplotlib
matplotlib.use('agg') 
# the automatic one on my comps (TkAgg) causes segfaults in certain cases, 
# see: https://github.com/matplotlib/matplotlib/issues/5795
import matplotlib.pyplot as plt
from time import process_time
import gzip
import subprocess
import pathos.multiprocessing as mp
import psutil

def inverse_dict(d,keys=None):
	res = {}
	keys = keys if not None is keys else d.keys()
	for k in keys:
		v = d[k]
		if not v in res:
			res[v] = [k]
		else:
			res[v].append(k)
	return res

def class_to_dict(class_instance,ignoring_list=None):
	if None is ignoring_list:
		ignoring_list = []
	values = [p for p in dir(class_instance) if (not "__" in p) and\
	 (not callable(getattr(class_instance,p))) and (not p in ignoring_list)]
	return {p:getattr(class_instance,p) for p in values}


class OhHeck(Exception):
	def __init__(self):
		pass

def do_timed(process_start_time,thing,thing_name="that",file=None):
	file = sys.stdout if None is file else file
	print(thing_name,"took:",process_time()-process_start_time,"seconds",flush=True,file=file)
	return thing

def chronological_scatter(vec,alpha=1,title="",vec2=None,vec1label=None,vec2label=None,filename=None,s=2):
	plt.scatter(range(len(vec)),vec,alpha=alpha,label=vec1label,s=s)
	if not None is vec2:
		plt.scatter(range(len(vec2)),vec2,alpha=alpha,label=vec2label,s=s)
	plt.title(title)
	if not (None is vec1label and None is vec2label):
		plt.legend()
	if not None is filename:
		prepare_directory(filename,includes_filename=True)
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()
  
def clean_val(num,digits=3):
	if digits == np.inf:
		return num
	if num in [np.inf,np.nan]:
		return num
	return round(num,digits)
	# return int(num*pow(10,digits))/pow(10,digits) 

def pick_index_from_distribution(vector): # all vector values must be >=0
	num = random.uniform(0,np.sum(vector))
	sums = np.cumsum(vector)
	for i in range(len(sums)):
		if num < sums[i]:
			return i
	#just in case
	return i

def prepare_directory(path,includes_filename=False):
	if includes_filename:
		path = "/".join(path.split("/")[:-1])
	if len(path)==0:
		return
	if not os.path.exists(path):
		# print("making path:",path)
		os.makedirs(path)

def overwrite_file(contents,filename,dont_zip=False):
	if not dont_zip:
		filename += "" if filename.endswith(".gz") else ".gz"
	prepare_directory(filename,includes_filename=True)
	open_fun = open if dont_zip else gzip.open
	with open_fun(filename,'wb') as f:
		pickle.dump(contents,f)

def load_from_file(filename,quiet=False):
	if not os.path.exists(filename):
		if not filename.endswith(".gz"):
			return load_from_file(filename+".gz",quiet=quiet) # maybe have it zipped
		if not quiet:
			print("no such file: ",filename)
		return None
	open_fun =  gzip.open if filename.endswith(".gz") else open 
	with open_fun(filename,'rb') as f:
		res = pickle.load(f)
	return res

def steal_attr(dest,source,attr):
	setattr(dest,attr,getattr(source,attr))

def things_in_path(path,ignoring_list=None,only_folders=False,only_files=False):
	if not os.path.exists(path):
		return []
	ignoring_list = ignoring_list if not None is ignoring_list else []
	ignoring_list.append(".DS_Store")
	ignoring_list.append(".ipynb_checkpoints")
	res = sorted([name for name in os.listdir(path) if not name in ignoring_list])
	if only_folders:
		res = [r for r in res if os.path.isdir(path+"/"+r)]
	if only_files:
		res = [r for r in res if os.path.isfile(path+"/"+r)]
	return res

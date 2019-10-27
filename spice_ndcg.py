# original file taken from SPiCe competition with credit as below. Have modified it somewhat 
# for my needs. Might not work for original SPiCe ndcg files.

# -*- coding: utf-8 -*-
"""
	Created in Mon July 41 10:47:00 2016
	
	@author: Remi Eyraud
	
	Usage: python score_computation.py rankings_file targets_file
	Role: compute the score of a set of rankings given the target next symbols probabilities
	Example: python score_computation.py 0.spice.ranking ../targets/0.spice.target.public
"""


from sys import *
import math
import string
import os

def find_proba(letter,target):
	for i in range(len(target)):
		if target[i]==letter:
			return float(target[i+1])
	return 0

def modified_score_rankings(rankings_file,targets_file): 
	# my additional format of target file: topmost line is just ndcg_k. if no such line in file, assume its spice, i.e. ndcg_k=5.
	with open(targets_file,"r") as f:
		firstline = f.readline().split()
		if len(firstline) == 1:
			ndcg_k = int(firstline[0])
			ditch_one_target_line = True 
		else:
			ndcg_k = 5
			ditch_one_target_line = False
	# print("using ndcg k:",ndcg_k,", and ditching first line:",ditch_one_target_line)
	r = open(rankings_file, "r") # format: 1 line per prefix, each line with 5 most likely tokens in order of decreasing likelihood
	t = open(targets_file, "r") # format given by SPiCe: total weight for 5 top, then all tokens. 

	if ditch_one_target_line:
		t.readline()

	score = 0
	nb_prefixes = 0
	for ts in t.readlines():
		nb_prefixes += 1
		rs = r.readline()
		target = ts.split()#string.split(ts)
		ranking = rs.split()#string.split(rs)
		denominator = float(target[0])
		prefix_score = 0
		# ndcg bit
		k=1
		for elmnt in ranking:
			if k == 1:
				seen = [elmnt]
				p = find_proba(elmnt,target)
				prefix_score += p/math.log(k+1,2)
			elif elmnt not in seen:
				p = find_proba(elmnt,target)
				prefix_score += p/math.log(k+1,2)
				seen = seen + [elmnt]
			k += 1
			if k > ndcg_k:
			   break
	#print(nb_prefixes, su)
		score += prefix_score/denominator
	ndcg_score = score/nb_prefixes
	r.close()
	t.close()
	return ndcg_score


def read_ndcg_prefix_file(filename):
	# print("loading from file:",filename)
	if not os.path.exists(filename):
		return None, None
	with open(filename,"r") as f:
		res = f.readlines()
	len_res, alpha_size = tuple(map(int,res[0].split())) 
	alphabet = list(range(alpha_size)) 
	res = res[1:] # first line has metadata, read above ^
	res = list(map(lambda x:x.split()[1:],res)) # first number in each line is just its length 
	res = list(map(lambda x:list(map(int,x)), res)) # input file had strings for characters, return to the numbers
	return res, alphabet

def make_spice_filename(file_number,set_type,wgy_name=None,target=False):
	assert set_type in ["train","test","validate"]
	main_folder = "spice/" if None is wgy_name else "wgy_data/"
	middle = ".spice." if None is wgy_name else ".wgy."
	specific = str(file_number) if None is wgy_name else wgy_name
	if "train" == set_type:
		return main_folder+"train/"+specific+middle+"train"
	else:
		file_ending = ".public" if "validate" == set_type else ".private"
		if not target:
			return main_folder+"prefixes/"+specific+middle+"prefix" + file_ending
		return main_folder + "targets/"+specific+middle+"target"+ file_ending



def load_spice_prefs(file_number,set_type,wgy_name=None):
	return read_ndcg_prefix_file(make_spice_filename(file_number,set_type,wgy_name))


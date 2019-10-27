import numpy as np
from time import time


def hankel(model,P,S,a=None,print_every_ten=False):
	main_start = time()
	# model = ModelMemory(model) if not isinstance(model,ModelMemory) else model # don't get slowed down by repeat queries
	if isinstance(P[0],list):
		a = [] if None is a else [a]
	else:
		a = "" if None is a else a
	res = np.zeros((len(P),len(S)))
	start = time()
	for i,p in enumerate(P):
		start = time()
		if hasattr(model,"weights_for_sequences_with_same_prefix"):
			res[i] = model.weights_for_sequences_with_same_prefix(p+a,S)
		else:
			res[i] = [model.weight(p+a+s) for s in S]
		end = time()
		# print("row",i,"took",end-start,"seconds")
		if print_every_ten and i%10 == 1:
			print("10 rows took",time()-start,"seconds")
			start = time()
	# print("full hankel with a:",a," took",time()-main_start,"seconds")

	return res

def hankel_as(model,P,S):
	# model = ModelMemory(model) if not isinstance(model,ModelMemory) else model # don't wait for hankel to do this, make a shared one and save time
	res = {}
	for a in model.input_alphabet:
		res[a] = hankel(model,P,S,a)
	return res

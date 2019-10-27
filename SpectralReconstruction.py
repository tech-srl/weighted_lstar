# implementations of the paper "Spectral Learning of Weighted Automata" by 
# Borja Balle, Xavier Carreras, Franco M Luque, Ariadna Quattoni
import numpy as np
from WFA import WFA
from Hankel import hankel_as, hankel
# from scipy.sparse import csc_matrix
# from sparsesvd import sparsesvd
from time import process_time
from LanguageModel import LanguageModel
import sys
from Helper_Functions import overwrite_file, load_from_file


	
def make_P_S(model,n_p,n_s,hard_stop=False,max_attempts=1e5,max_sample_length=np.inf):
	model = LanguageModel(model)
	P = set()
	S = set()
	attempts = 0
	while (len(P)<n_p or len(S)<n_s) and attempts<max_attempts:
		attempts += 1
		w = model.sample(cutoff=max_sample_length)
		if len(P)<n_p:  # to keep from making P bigger than expected when n_s>n_p
			for i in range(len(w)+1):
				P.add(tuple(w[:i]))
				if hard_stop and len(P)>=n_p:
					break
		if len(S)<n_s:   # same idea, when n_p>n_s
			for i in range(len(w),-1,-1):
				S.add(tuple(w[i:]))
				if hard_stop and len(S)>=n_s:
					break
	P = sorted([list(w) for w in P],key=len)
	S = sorted([list(w) for w in S],key=len)
	if len(P)<n_p or len(S)<n_s:
		print("attempted",max_attempts,"samples, each of max length",max_sample_length,", could not get enough P/S but cutting short")
	return P,S


def trim_svd(U,d,V_t,k):
	return U[:,:k],d[:k],V_t[:k,:]

def their_algorithm(stuff,k):
	U,d,V_t = trim_svd(*(stuff["svd"]),k)
	alpha_t = stuff["h_s"] @ V_t.transpose()
	beta = np.linalg.pinv(stuff["h"] @ V_t.transpose()) @ stuff["h_p"]
	A = {}
	for a in stuff["alphabet"]:
		A[a] = np.linalg.pinv(stuff["h"]@V_t.transpose())@stuff["h_a"][a]@V_t.transpose()
	return WFA(alpha_t,beta,A)




def make_hankel_stuff(model,P,S,print_file):
	f = print_file
	start = process_time()
	stuff = {"alphabet":model.input_alphabet}
	stuff["h"] = hankel(model,P,S)
	# stuff["h_a"] = hankel_as(model,P,S)
	print("made the main hankel, that took:",int(process_time()-start),flush=True,file=f)
	stuff["h_a"] = {}
	for i,a in enumerate(stuff["alphabet"]):
		mini_start = process_time()
		stuff["h_a"][a] = hankel(model,P,S,a)
		print("made hankel for:",a,"(",i+1,"of",len(stuff["alphabet"]),"), that took:",int(process_time()-mini_start),file=f,flush=True)
	stuff["h_s"] = np.reshape(np.array([model.weight(s) for s in S]),(1,len(S)))
	stuff["h_p"] = np.reshape(np.array([model.weight(p) for p in P]),(len(P),1))

	stuff["hankel_time"] = process_time() - start
	svdstart = process_time()
	stuff["svd"] = np.linalg.svd(stuff["h"])
	stuff["svd_time"] = process_time() - svdstart
	stuff["rank"] = np.linalg.matrix_rank(stuff["h"])
	return stuff


def spectral_reconstruct(model,P,S,k_list,ready_hankel_things=None,print_file=None):
	f = print_file if not None is print_file else sys.stdout

	#make sure they start with the empty sequence
	assert len(P[0])==0
	assert len(S[0])==0
	model = LanguageModel(model)
	print("making spectral with P,S sizes:",len(P),len(S),file=f,flush=True)
	if None is ready_hankel_things:
		stuff = make_hankel_stuff(model,P,S,f)
	else:
		stuff = ready_hankel_things

	results = []
	total_times = []
	done_max = False
	for k in sorted(k_list):
		if k>=stuff["rank"]:
			if done_max:
				print("skipping",k,"onwards",file=f)
				break
			print("maxed out at",k,"so using k=rank=",stuff["rank"],file=f)  # this allows using a k that is 'greater' than the rank,
			# which is important in the case the exact rank is missed (eg if the rank is 15 but its just 
			# checking k=10,20,30, it will still do 20 but then skip 30)
			k = stuff["rank"] # don't make something higher than there actually is, making a WFA that thinks it has eg 5 states when it really has 2
			done_max = True
		start = process_time()
		results.append(their_algorithm(stuff,k))
		total_times.append(stuff["hankel_time"]+stuff["svd_time"]+process_time()-start)
	return results, total_times, stuff["hankel_time"], stuff["svd_time"], stuff




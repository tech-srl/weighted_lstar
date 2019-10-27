import numpy as np
from time import time, process_time
import os
from LanguageModel import LanguageModel

class WFA:
	def __init__(self,alpha,beta,A,is_lists=None,add_beta_tilde=True,source_dict=None,informal_name=None):
		if not None is source_dict:
			self._build_from_dict(source_dict)
			return
		self.n = len(beta)
		self.name = str(int(1e5*time()))+"_WFA_"+str(self.n) # use time to make sure to separate them and also list them in reasonable older in folders when looking
		self.informal_name = informal_name if not None is informal_name else "WFA"
		self.alpha = np.reshape(np.array(alpha),(1,self.n))
		self.beta = np.reshape(np.array(beta),(self.n,1))
		self.A = A
		self.input_alphabet = tuple(self.A.keys())
		self.is_lists = is_lists if not None is is_lists else (True in [isinstance(a,list) for a in self.input_alphabet])
		self.end_token = "<EOS>"
		self.internal_alphabet = self.input_alphabet + (self.end_token,)
		self.char2int = {a:i for i,a in enumerate(self.internal_alphabet)}
		self.has_beta_tilde = False
		if add_beta_tilde:
			try_to_add_beta_tilde(self)

	def _build_from_dict(self,source):
		[setattr(self,n,source[n]) for n in source]

	def initial_state(self):
		return self.alpha

	def next_state(self,s,t):
		return s@self.A[t]

	def _state_from_pref(self,pref,initial_state=None):
		s = initial_state if not None is initial_state else self.initial_state()
		for a in pref:
			s = s@self.A[a]
		return s

	def _state_weight(self,s,as_prefix=False):
		if as_prefix and not self.has_beta_tilde:
			print("cannot compute prefix weight: dont have beta tilde")
			return
		beta = self.beta_tilde if as_prefix else self.beta
		r = s@beta
		return r[0][0]

	def weight(self,w,as_prefix=False,state=None):
		s = self._state_from_pref(w,initial_state=state)
		return self._state_weight(s,as_prefix=as_prefix)

	def state_probs_dist(self,state):
		next_tokens_states = [self.next_state(state,t) for t in self.input_alphabet]
		next_tokens_prefix_probs = [self._state_weight(s,as_prefix=True) for s in next_tokens_states]
		stopping_prob = self._state_weight(state,as_prefix=False)
		prefix_probs = next_tokens_prefix_probs + [stopping_prob] # before normalisation, this is the total probability held in each continuation from this state
		return (np.array(prefix_probs)/self._state_weight(state,as_prefix=True)).tolist()

	def state_char_prob(self,state,char):
		pref_weight = self._state_weight(state,as_prefix=True)
		if pref_weight == 0:
			return -1
		if char == self.end_token:
			return self._state_weight(state,as_prefix=False)/pref_weight
		with_token_weight = self._state_weight(self.next_state(state,char),as_prefix=True)
		return with_token_weight/pref_weight


#####
# code for converting to wfa that gives prefix probabilities
# from paper "Spectral Learning of Weighted Automata"
# by Borja Balle, Xavier Carreras, Franco M Luque, Ariadna Quattoni
#####
def singular_values(A):
	return np.linalg.svd(A)[1] 

def schatten_norm(A,p):
	return np.linalg.norm(singular_values(A),ord=p)
	# wikipedia notes all schatten norms are submultiplicative, 
	# though it is not clear if p<=0 is still a schatten norm (nothing is said about positivity of p,
	# and numpy documentation suggests it is fine)
	# also note singular values are real and non negative so applying p-norm is fine and makes sense
	

def schatten_norm_func(p):
	return lambda x:schatten_norm(x,p)
	
def bigA(wfa):
	sigma = wfa.input_alphabet
	A = wfa.A[sigma[0]]
	for a in sigma[1:]:
		A = A + wfa.A[a]
	return A

def definitely_irredundant(wfa): 
	# definition is that there exists *some* submultiplicative matrix norm for which the value is <1    
	A = bigA(wfa)
	submultiplicative_matrix_norms = [schatten_norm_func(p) for p in range(20)] # only ones i know atm, can add more. can also increase range of p arbitrarily 
	# turns out frobenius norm is just schatten norm for p=2 so not explicitly adding i
	for f in submultiplicative_matrix_norms:
		if f(A) < 1:
			return True
	return False
	
def spectral_radius(A):
	return np.max([np.abs(a) for a in np.linalg.eig(A)[0]])

def definitely_not_irredundant(wfa):
	A = bigA(wfa)
	if spectral_radius(A) >= 1:
		return True 
	# definition notes that neccessary condition for irredundancy is spectral radius < 1
	return False


# from WFA import definitely_not_irredundant, definitely_irredundant, bigA # when you want to copy this to the notebook
def probably_irredundant(wfa): # at least for purposes of the quality they relied on
	start = process_time()
	# print("checking irredundancy",flush=True)
	def_not = definitely_not_irredundant(wfa)
	# print("checking definitely not irredundant took:",time()-start,"result was:",def_not,flush=True)
	if def_not:
		return False
	start = process_time()
	def_yes = definitely_irredundant(wfa)
	# print("checking definitely irredundant took:",time()-start,"result was:",def_yes,flush=True)
	if def_yes:
		return True
	# print("didn't get a definite result, going into full check",flush=True)
	# start = time()
	A = bigA(wfa)
	jumps = 100 # move things 100 times faster than straightforward x=(x@A)+A^0
	base = np.zeros(A.shape)
	for i in range(jumps): # i=0,1,..,jumps-1
		base += np.linalg.matrix_power(A,i)
	mult = np.linalg.matrix_power(A,jumps)
	L = np.linalg.inv(np.eye(len(A))-A)
	running_sum = base
	under_threshold_count = 0
	threshold = 1e-3 # honestly like numerical errors can get pretty bad for big matrices soooo... this is probably good enough. remember they start off thousands away
	for i in range(1000000):
		# if i%5000 == 0:
			# print_mat_clean(running_sum-L)
			# print(np.sum(abs(running_sum-L)))
		running_sum = (running_sum @ mult) + base
		diff = np.sum(abs(running_sum-L))
		if diff == np.inf or np.isnan(diff):
			print("A^0+A+A^2+... went to",diff,",that took:",process_time()-start,"seconds",flush=True)
			return False
		if diff < threshold:
			under_threshold_count += 1
			# print(under_threshold_count)
			if under_threshold_count >=1000:
				# print("seem to have converged, that took:",time()-start,"seconds",flush=True)
				return True
		else:
			under_threshold_count = 0
	# print("full check neither converged nor diverged. That took:",time()-start,"seconds",flush=True)
	return False

		
def try_to_add_beta_tilde(wfa,force=True): # using lemma 1
	if not force:
		wfa.probably_irredundant = probably_irredundant(wfa)
		if not wfa.probably_irredundant:
			print("sum of A^k doesn't seem to converge to (I-A)^(-1), so can't do prefix probabilities")
			return
	# else:
		# print("not checking irredundancy bc gonna have to do the manip anyway")
	A = bigA(wfa)
	beta_tilde = (np.linalg.inv(np.eye(len(A))-A) @ wfa.beta)   
	# btw, if the wfa is not only irredundant but actually deterministic & probabilistic, 
	# then beta_tilde will just be all ones (the weight of a prefix is just the weight 
	# of all the transitions on it so far: the sum of the weights of all of its 
	# continuations will be one b/c deterministic probabilistic), so don't freak out if that seems to happen a lot
	wfa.beta_tilde = beta_tilde
	wfa.has_beta_tilde = True



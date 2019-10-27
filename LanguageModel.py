from Helper_Functions import pick_index_from_distribution, steal_attr, OhHeck, prepare_directory
import numpy as np
from time import time
import os
import random

wgy_ndcg_k = 2 # have to think of better place/way to store this

class LanguageModel:
	def __init__(self,model):
		for attr in ["initial_state","next_state","state_probs_dist","state_char_prob"]:
			if not hasattr(model,attr) and callable(getattr(model,attr)):
				print("model",str(model)[:30],"is missing attribute:",attr)
				raise OhHeck()
		self.model = model
		[steal_attr(self,model,attr) for attr in ["input_alphabet","end_token","internal_alphabet","name","informal_name"]]
		self.int2char = self.internal_alphabet # its a list from def of internal_alphabet
		self.char2int = {c:i for i,c in enumerate(self.int2char)}

	def initial_state(self):
		return self.model.initial_state()

	def next_state(self,state,char):
		return self.model.next_state(state,char)

########## 
# theoretically, of _state_probs_dist, _state_char_prob, and weight, only really need one. 
# but efficient implementation for them may vary greatly: 
# eg, for WFA, next-token probability is computed by 2 weight calculations: 
# one for pref and one for pref+char, with weight being computed directly without considering
# conditional probs. conversely, in DPWFA and RNN, weight is product of conditional probs.
# to force this behaviour on a WFA would be woefully inefficient.
# forcing the opposite behaviour on DPWFA or RNN is also not great: why compute 2 whole sequence 
# weights just to get a conditional probability when that is already available?
# tl:dr - let the model make its own decisions on how to implement these 3
##########
	def state_probs_dist(self,state): # next token weights for each of internal alphabet, in model's internal alphabet order
		return self.model.state_probs_dist(state)

	def state_char_prob(self,state,char):
		return self.model.state_char_prob(state,char)

	def weight(self,sequence,as_prefix=False,state=None):
		if hasattr(self.model,"weight"): # rnns, wfas may prefer their own implementation
			return self.model.weight(sequence,as_prefix=as_prefix,state=state) 
		
		# for now - on your own head be it if you dont pass this assert, cause not gonna waste time on it
		# assert not self.end_token in sequence # avoid trouble, especially as self.probability_of_sequence_after_state is fine with it, at least as the last token


		s = self.model.initial_state() if None is state else state
		res = self.probability_of_sequence_after_state(s,sequence)
		if not as_prefix:
			res *= self.state_char_prob(s,self.end_token)
		return res

	def probability_of_sequence_after_state(self,state,seq):
		p = 1
		for c in seq[:-1]: # get to the last state, counting probs along the way
			p *= self.state_char_prob(state,c)
			state = self.next_state(state,c)
		if seq: # add the last probability. make sure there is one to avoid crashing on empty seqs
			p*= self.state_char_prob(state,seq[-1]) # separation of last token from others is so \
			# that dont also attempt self.next_state on it, as it might be EOS eg when being called from the learner for separating suffixes

		return p
		
	def _append_token(self,pref,c):
		if isinstance(pref,list):
			return pref + [c]
		if isinstance(pref,tuple):
			return pref + (c,)
		return pref + c

	def sample(self,from_seq=None,cutoff=np.inf,avg_cutoff=np.inf,empty_sequence = None):
		def token_from_state(state):
			return self.int2char[pick_index_from_distribution(self.state_probs_dist(state))]
		def early_stop():
			return (early_stopping_prob>0) and random.random() < early_stopping_prob
		res = empty_sequence if not None is empty_sequence else () # default to making list, but could also make tuple or string if requested
		res = from_seq if not None is from_seq else res

		early_stopping_prob = 1/avg_cutoff
		s = self._state_from_sequence(res)
		while True:
			c = token_from_state(s)
			if (c == self.end_token and avg_cutoff==np.inf) or len(res)>=cutoff or early_stop(): # reached end of sample and weren't trying to get a certain avg length OR got too long OR random cutoff for random length samples
				return res
			if c == self.end_token: # got end token but have to keep going b/c were looking for some kind of sample
				for _ in range(100): # not trying more than that to get another one really
					c = token_from_state(s)
					if not c == self.end_token:
						break
				if c == self.end_token: # give up trying to keep going
					return res
			res = self._append_token(res,c)
			s = self.next_state(s,c)

	def _state_from_sequence(self,sequence,s=None):
		if None is s:        
			s = self.initial_state()
		for c in sequence:
			s = self.next_state(s,c)
		return s

	def distribution_from_sequence(self,sequence):
		return {c:p for c,p in zip(self.int2char,self.state_probs_dist(self._state_from_sequence(sequence)))}

	def _most_likely_token_from_state(self,state,k=1):
		if 1==k:
			return self.int2char[np.argmax(self.state_probs_dist(state))]
		a = np.array(self.state_probs_dist(state)) # some models give np arrays, some dont..
		relevant = np.argpartition(a,-k)[-k:]
		decreasing_order = relevant[np.argsort(-a[relevant])] #a is positive, and this sorts in increasing order
		return [self.int2char[i] for i in decreasing_order]

	def errors_in_group(self,group,s=None): # definitely check against results of above function before using
		if len(group)==0:
			return 0
		if None is s:  
			s = self.initial_state()
		by_tokens = split_by_first(group,self.end_token,self.input_alphabet)
		errors = len(group) - len(by_tokens[self._most_likely_token_from_state(s)])
		for t in self.input_alphabet: # check more errors for sequences that are still going
			if len(by_tokens[t])==0:
				continue #don't waste time computing next state or upset dynet with even more nodes than need be
			errors += self.errors_in_group(by_tokens[t],self.next_state(s,t))
		return errors

	def next_token_preds(self,sequences,s=None,pref=None,res=None):
		sequences = quickfix(sequences) # its gonna be tuples!
		if None is res:
			res = {}
		if not sequences:
			return res
		if None is s:
			s = self.initial_state()
		if None is pref:
			pref = sequences[0][:0]					# doing dumb hack with tuples for this one so don't 
		if 0 in [len(seq) for seq in sequences]: 	# use original empty sequence, use whatever input you're getting...
			res[pref] = self._most_likely_token_from_state(s)
		by_tokens = split_by_first(sequences,self.end_token,self.input_alphabet)
		for t in self.input_alphabet:
			if not by_tokens[t]: #empty
				continue
			res = self.next_token_preds(by_tokens[t],s=self.next_state(s,t),\
				pref=self._append_token(pref,t),res=res)
		return res

	def weights_for_sequences_with_same_prefix(self,prefix,suffixes):
		if hasattr(self.model,"weights_for_sequences_with_same_prefix"):
			return self.model.weights_for_sequences_with_same_prefix(prefix,suffixes)
			# rnns lose accuracy when made to convert to val @ every step instead
			# of using expression and only evaluating it at the end. allow
			# model to use its own implementation of this if it has one
		def _compute_weight(base_state,lookaheads,seq):
			if len(seq)<1:
				return (initial_weight*self.weight(seq,state=base_state))
			base_state,base_weight = lookaheads[seq[0]]
			return (base_weight*self.weight(seq[1:],state=base_state))
		s = self.initial_state()
		res = 1
		for c in prefix:
			res *= self.state_char_prob(s,c)
			s = self.next_state(s,c)
		initial_weight = res
		lookaheads = {t:(self.next_state(s,t),initial_weight*self.state_char_prob(s,t)) \
						for t in self.input_alphabet}
		return [_compute_weight(s,lookaheads,seq) for seq in suffixes]
		# return [(initial_weight*self.weight(seq,base_state=s)).value() for seq in suffixes]

	def all_sequence_transitions(self,sequence,including_stop=False,prefix=None):
		prefix = prefix if not None is prefix else []
		s = self._state_from_sequence(prefix)
		res = []
		for c in sequence:
			res.append(self.state_char_prob(s,c))
			s = self.next_state(s,c)
		if including_stop:
			res.append(self.state_char_prob(s,self.end_token))
		return res

	def last_token_probabilities_after_pref(self,pref,suffixes): 
		base_state = self._state_from_sequence(pref)
		res = []
		for sequence in suffixes:
			if len(sequence)==0:
				res.append(1) # default value for empty string continuation
				continue
			s = self._state_from_sequence(sequence[:-1],s=base_state)
			res.append(self.state_char_prob(s,sequence[-1]))
		return res

	def probability_of_char_after_prefix(self,prefix,char):
		return self.state_char_prob(self._state_from_sequence(prefix),char)

	def probability_of_ending_after_prefix(self,prefix): 
		return self.probability_of_char_after_prefix(prefix,self.end_token)

	def WER(self,sequences,gold=None,gold_dict=None):
		if None is gold and None is gold_dict: # WER against the sequences themselves
			sum_errors = self.errors_in_group(sequences)
			sum_lengths = sum(len(s)+1 for s in sequences) #has to predict every token and then also the end
			return sum_errors/sum_lengths
		pref_counts = make_prefs_count_dict(quickfix(sequences))
		all_prefs = list(pref_counts.keys())
		if None is gold_dict:
			gold = LanguageModel(gold)
			assert self.end_token == gold.end_token
			gold_dict = gold.next_token_preds(all_prefs)
		# else: # just hope for the best about the end token thing frankly idk. check it outside with your original model or something
		self_dict = self.next_token_preds(all_prefs)
		errors = sum(pref_counts[p] for p in all_prefs if not self_dict[p]==gold_dict[p])
		num_preds = sum((len(p)+1) for p in sequences)
		return errors/num_preds


	# def perplexity(self,sequences): # could probably do all this a lot faster 
	# # with a couple of one liners but its infinity so often that its better 
	# # to go this way and let it cut off early when it can
	# 	num_skipped = 0
	# 	num_observations = np.sum(len(s)+1 for s in sequences)
	# 	total_logs = 0
	# 	for s in sequences:
	# 		w = one_sequence_logs(self,s)
	# 		if -np.inf == w:
	# 			return np.inf # early stop when nothing to do, which just happens a lot
	# 		else:
	# 			total_logs += w
	# 	if total_logs == -np.inf:
	# 		return np.inf

	# 	res = np.inf if num_observations == 0 else np.power(2,-total_logs/num_observations)
	# 	return res

	def make_spice_preds(self,prefixes,filename=None):
		if None is filename:
			filename = "temporary_preds_"+str(time())+".txt"
		assert not None in prefixes
		prepare_directory(filename,includes_filename=True)
		with open(filename,"w") as f:
			for p in prefixes:
				state = self._state_from_sequence(p)
				preds = self._most_likely_token_from_state(state,k=len(self.internal_alphabet)) # just list them all in decreasing order
				preds = [str(t) if not t == self.model.end_token else "-1" for t in preds]
				f.write(" ".join(preds)+"\n")
		return filename		

def one_sequence_logs(model,seq):
	transitions = model.all_sequence_transitions(seq,including_stop=True)
	if True in [t<=0 for t in transitions]:
		return -np.inf
	return np.sum(np.log2(v) for v in transitions)


def quickfix(sequences):
	if isinstance(sequences[0],list):
		sequences = [tuple(l) for l in sequences]
	return sequences

def make_prefs_count_dict(sequences):
	res = {}
	for s in sequences:
		for i in range(len(s)+1):
			pref = s[:i]
			if pref in res:
				res[pref] += 1
			else:
				res[pref] = 1
	return res
	# this is very slow:
	# all_prefixes = [p[:i] for p in sequences for i in range(len(p)+1)]
	# pref_counts = {p:all_prefixes.count(p) for p in set(all_prefixes)}

def split_by_first(group,end_token,alphabet):
	by_tokens = {end_token:[None for s in group if len(s)==0]}  # seems silly but helps deal with EOS with less ifs
	group = [s for s in group if len(s)>0]
	for t in alphabet:
		by_tokens[t] = [s[1:] for s in group if (s[0]==t)]
	return by_tokens

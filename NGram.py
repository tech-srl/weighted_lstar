from time import process_time

class NGram:
	def __init__(self, n, alphabet, sequences,source_dict=None,informal_name=None, too_big_dont_make=2e6): # still assume sequences aren't tuples...
		if not None is source_dict:
			self._build_from_dict(source_dict)
			return
		start = process_time()
		self.is_lists = isinstance(sequences[0],list)
		self.name = str(n)+"-gram_from_"+str(len(sequences))# honestly don't expect repetition for these guys so can use simple name
		self.informal_name = informal_name if not None is informal_name else self.name # name is already pretty informal tbh
		self.end_token = "<EOS>"
		self.input_alphabet = tuple(alphabet)
		self.internal_alphabet = self.input_alphabet + (self.end_token,)
		self.int2char = self.internal_alphabet
		self.char2int = {c:i for i,c in enumerate(self.int2char)}
		self.n = n
		self.unseen_distribution = [1/len(self.internal_alphabet) for a in self.internal_alphabet]

		sequences = [tuple(s) for s in sequences] # we're working with tuples only
		sequences = [(self.end_token,)*(n-1) + s + (self.end_token,) for s in sequences] # padding on all sides

		self._state_probs_dist = {}
		self.successfully_initiated = False
		num_prefs = 0
		for s in sequences:
			for i in range(len(s)+1-self.n):
				pref = s[i:i+self.n-1]
				next_token = s[i+self.n-1]
				if not pref in self._state_probs_dist:
					self._state_probs_dist[pref] = {c:0 for c in self.internal_alphabet}
					num_prefs += 1
					if num_prefs > too_big_dont_make:
						return
				self._state_probs_dist[pref][next_token] += 1
		print("observed",len(self._state_probs_dist),"possible prefixes")
		for p in self._state_probs_dist:
			preds = self._state_probs_dist[p]
			total = sum([preds[a] for a in self.internal_alphabet]) # total is non-zero bc p was only made if it exists in sequences
			preds = {a:preds[a]/total for a in self.internal_alphabet}
			self._state_probs_dist[p] = preds
		self.init_from_sequence_time = process_time() - start
		self.successfully_initiated = True

	def _build_from_dict(self,source):
		[setattr(self,n,source[n]) for n in source]

	def initial_state(self):
		return (self.end_token,)*(self.n-1)

	def next_state(self,state,char):
		return (state+(char,))[1:] #first add last char then take away first, to correctly handle case when state is empty (tho tbh can also handle that with an if)

	def state_hash(self,state):
		return state

	def state_probs_dist(self,state):
		if not state in self._state_probs_dist:
			return self.unseen_distribution
		return [self._state_probs_dist[state][a] for a in self.internal_alphabet]

	def state_char_prob(self,state,char):
		return self.state_probs_dist(state)[self.char2int[char]]

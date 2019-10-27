import numpy as np
from Helper_Functions import OhHeck, clean_val
from time import time, process_time
from IPython.display import display
import os


class PDFA: #deterministic probabilistic WFA that just has the states listed out like a DFA. helper class for specific WFA things
# warning: might have unreachable states!
	def __init__(self,informal_name=None,transitions_and_weights=None,
			end_token="EOS",initial_state=None):
		self.informal_name = informal_name if not None is informal_name else "PDFA"
		if not None is transitions_and_weights:
			transitions,transition_weights = transitions_and_weights
		alphabet = tuple(transitions[0].keys())
		self._boring_init_parts(alphabet,end_token,initial_state)
		self.transitions = transitions
		self.transition_weights = transition_weights 
		for s in self.transition_weights:
			self.transition_weights[s][self.end_token] = 1 - sum([self.transition_weights[s][a] for a in alphabet])
		self.probabilities_list = {}
		for s in self.transition_weights:
			self.probabilities_list[s] = [self.transition_weights[s][a] for a in self.internal_alphabet]
		self._add_not_a_state_transitions()
		self.check_reachable_states()
		self.normalise() # numerical errors......


	def _add_not_a_state_transitions(self):
		default_probability = 1/len(self.internal_alphabet) # uniform probability over all tokens and ending
		self.transition_weights[self.not_a_state]={a:default_probability for a in self.internal_alphabet}
		self.transitions[self.not_a_state]={a:self.not_a_state for a in self.internal_alphabet}
		self.probabilities_list[self.not_a_state]=[default_probability for a in self.internal_alphabet]

	def draw_nicely(self,max_size=60,precision=3,filename=None,keep=False,dpi=100,add_state_name=False): # probabilities_list, transition_weights, transitions
		import graphviz as gv
		import functools
		from IPython.display import Image
		digraph = functools.partial(gv.Digraph, format='png')
		g = digraph()
		g.attr(dpi=str(dpi))
		states = list(self.check_reachable_states())
		if len(states)>max_size:
			return
		def state_label(s):
			w = self.transition_weights[s][self.end_token]
			label = str(clean_val(self.transition_weights[s][self.end_token],precision))
			if add_state_name:
				label = str(s)+" : "+label
			return label
		def state_shape(s):
			return 'hexagon' if s == self.initial_state() else 'circle'
		def transition_label(s,a):
			return str(a)+", "+str(clean_val(self.transition_weights[s][a],precision))
		[g.node(str(s),label=state_label(s),shape=state_shape(s)) for s in states if not s==self.not_a_state] # don't draw non state

		for s in states:
			for a in self.input_alphabet:
				if self.transition_weights[s][a]>0: # don't draw trash
					g.edge(str(s),str(self.transitions[s][a]),label=transition_label(s,a))

		filename = 'img/automaton'+str(time()) if None is filename else filename
		img_filename = g.render(filename=filename)
		display(Image(filename=img_filename))
		os.remove(filename)     # definitely spam
		if not keep:
			os.remove(img_filename) # only spam if i didnt ask to keep it

	def initial_state(self):
		return self._initial_state

	def next_state(self,state,char):
		return self.transitions[state][char]

	def state_hash(self,state):
		return state

	def state_probs_dist(self,state):
		return self.probabilities_list[state]

	def state_char_prob(self,state,char):
		return self.transition_weights[state][char]

	def weight(self,w,as_prefix=False,state=None):
		s = self.initial_state() if None is state else state
		res = 1
		for a in w:
			res *= self.state_char_prob(s,a)
			s = self.next_state(s,a)
		if not as_prefix:
			res *= self.state_char_prob(s,self.end_token)
		return res

	def _boring_init_parts(self,alphabet,end_token,initial_state):
		self.name = str(int(time()*1000)) # these happen real fast and write over each other if you just round to seconds
		assert not None in [end_token,initial_state], "end token: "+str(end_token)+" initial: "+str(initial_state)
		self._initial_state = initial_state
		self.not_a_state = -1
		self.end_token = end_token
		self.input_alphabet = alphabet
		self.internal_alphabet = tuple(self.input_alphabet) + (self.end_token,)
		self.char2int = {a:i for i,a in enumerate(self.internal_alphabet)}

	def normalise(self):
		for i in self.transition_weights:
			total = np.sum(self.probabilities_list[i])
			if np.abs(total-1)>1e-6:
				print("weights dont seem to add up to 1 at every state! (add up to:",total,")")
				raise OhHeck()
			self.probabilities_list[i] = [p/total for p in self.probabilities_list[i]]
			self.transition_weights[i] = {a:self.transition_weights[i][a]/total for a in self.transition_weights[i]}

	def check_reachable_states(self): # this does not include states reachable with 0 transitions, which are only here for completeness
		new_states = [self.initial_state()]
		explored_states = set()
		self.depth = 0
		while new_states: # not empty
			explored_states.update(new_states)
			new_states = list(set([self.transitions[s][a] for a in self.input_alphabet for s in new_states if self.transition_weights[s][a]>0])) # DONOT skimp on the squishing to a set here or you WILL get an exponentially growing number of new states to deal with and this function will basically grind to a halt
			new_states = [s for s in new_states if not s in explored_states]
			self.depth += 1
		self.num_reachable_states = len(explored_states)
		self.n = self.num_reachable_states # idk i think something somewhere expects this
		# print("pfa size is",self.num_reachable_states,", has depth:",self.depth)
		return explored_states


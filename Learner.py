import numpy as np
from Helper_Functions import OhHeck, clean_val, inverse_dict,  do_timed, overwrite_file
from PDFA import PDFA
from time import process_time
from heapq import heappush,heappop
from KDTree import KDTree
from LanguageModel import LanguageModel
from scipy.sparse import lil_matrix
from sklearn.cluster import DBSCAN
from itertools import combinations
import sys, traceback
# TODO: remove the whole init-from-learner bit in PDFA, not gonna happen any more (gonna give it its transitions and weights directly)

full_match_str = "full"
partial_match_str = "partial"
nothing_str = "nothing"

def tup2seq(tup):
	return str(tup) # changed bc confusing when with multiple-character tokens
	# return ''.join([str(t) for t in tup]) 

class Table:
	def __init__(self,target,max_P,max_S,atol,interval_width,prints_path,\
		s_separating_threshold,expanding_time_limit,\
		progress_P_print_rate,interesting_p_transition_threshold,very_verbose):
		self.prints_path = prints_path
		self.P = [()]
		self.S = [(t,) for t in target.internal_alphabet] # always add to end of S!
		self.target = target
		self.max_P = max_P
		self.max_S = max_S
		self.expanding_time_limit = expanding_time_limit
		self.table_start = process_time()
		self.atol = atol
		self.interval_width = interval_width
		self.prefix_weights_dict = {} # cache
		self.prefix_rows = {} # cache
		self.s_separating_threshold = s_separating_threshold
		self.interesting_p_transition_threshold = interesting_p_transition_threshold
		self.number_ignored_suffixes_in_last_expand = 0
		self.compared_log = {}
		self.last_suffix_add_time = process_time()
		self.progress_P_print_rate = progress_P_print_rate
		self.skipped_P_count = 0
		self.very_verbose = very_verbose

	def compared_so_far(self,p1,p2):
		return max(self.compared_log.get((p1,p2),0),self.compared_log.get((p2,p1),0))

	def note_compared(self,p1,p2):
		self.compared_log[(p1,p2)] = len(self.S)

	def equal(self,r1,r2):
		return np.allclose(r1,r2,atol=self.atol)

	def prefix_to_nprow(self,prefix):
		r = self.prefix_rows.get(prefix,np.array([]))
		remaining_S = self.S[len(r):]
		if remaining_S:
			remaining = self.target.last_token_probabilities_after_pref(prefix,remaining_S)
			r = np.array(r.tolist() + remaining)
			self.prefix_rows[prefix] = r
		return r

	def get_matching_ps(self,row):
		close = self.prefix_tree.get_all_close(row,self.atol)
		close = [p for p in close if self.equal(row,self.prefix_to_nprow(p))]
		return close

	def prefix_then_suffix_prob(self,prefix,suffix):
		# for now - not bothering to remember states that are prolly gonna be reused a lot tbfh
		s = self.target._state_from_sequence(prefix)
		return self.target.probability_of_sequence_after_state(s,suffix)

	def most_influential_separating_suffix(self,p_main,p_close,all_conts,all_close_conts,suffixes):
		relevant = []
		for r1,r2,t in zip(all_conts,all_close_conts,self.target.input_alphabet):
			if self.equal(r1,r2):
				continue
			for v1,v2,s in zip(r1,r2,suffixes):
				if not self.equal(np.array([v1]),np.array([v2])):
					main_prob = self.prefix_then_suffix_prob(p_main,(t,)+s)
					p_close_prob = self.prefix_then_suffix_prob(p_close,(t,)+s)
					min_prob = min(main_prob,p_close_prob) # i.e. (t,)+s differentiate prefixes p1 and p2, and happens with probability at least min_prob after each one 
					# (we care about the minimum probability after p1 and p2 because e.g. if the probability of (t,)+s happening after p1 is 0
					# then it is not an interesting separating suffix, even if its probability after p2 is high
					relevant.append((t,s,min_prob))
		# print("number of potential separating suffixes for",p_main,"and",p_close,":",len(relevant),file=self.prints_path)
		# print("they are\n:","\n".join([str(x) for x in sorted(relevant,key=lambda x:x[2],reverse=True)]),file=self.prints_path)
		if not relevant:
			return None,None
		most_relevant =  relevant[np.argmax([x[2] for x in relevant])] # tuple with highest min_prob
		# print("most relevant was:",most_relevant,"with conditional probability:",most_relevant[2],file=self.prints_path)
		return (most_relevant[0],)+most_relevant[1],most_relevant[2]
		
	def check_consistency(self,prefix): # remember for each p1,p2 up to which index in S they've already been checked
		row = self.prefix_to_nprow(prefix)
		close_ps = self.get_matching_ps(row)
		all_conts_full_S = [self.prefix_to_nprow(prefix+(t,)) for t in self.target.input_alphabet]
		close_p_weights = [self.prefix_weight(p) for p in close_ps]
		num_checks = 1
		for _,close_p in sorted(zip(close_p_weights,close_ps),key=lambda x:x[0],reverse=True):
			if close_p == prefix:
				continue # don't waste time # TODO: havent actually run code since adding this
			start = process_time()
			num_checks += 1
			all_close_conts = [self.prefix_to_nprow(close_p+(t,)) for t in self.target.input_alphabet] # todo: these should also be sorted by max_(t in alphabet)(min_(p\in main_p,close_p)(likelihood of t appearing after p))
			checked_so_far = self.compared_so_far(prefix,close_p)
			all_close_conts = [r[checked_so_far:] for r in all_close_conts] # next-one-token vectors for prefix that is similar to current on current S
			all_conts = [r[checked_so_far:] for r in all_conts_full_S] # prefix vectors
			suffixes = self.S[checked_so_far:]
			new_suffix,new_suffix_relevance = self.most_influential_separating_suffix(prefix,close_p,all_conts,all_close_conts,suffixes)
			self.note_compared(prefix,close_p) #will now process the results of the comparison, but jot down that it never has to be done again (on this part of S)
			if not None is new_suffix:
				assert not new_suffix in self.S # else wtf	
				if new_suffix_relevance > self.s_separating_threshold: 
					self.S.append(new_suffix)
					print("added separating suffix:",tup2seq(new_suffix),file=self.prints_path)
					print("time since last suffix add:",process_time()-self.last_suffix_add_time,file=self.prints_path)
					self.last_suffix_add_time = process_time()
					print("overall ignored",self.number_ignored_suffixes_in_last_expand,"suffixes so far in this expand",file=self.prints_path,flush=True)
					return False
				else:
					print("best separating suffix",new_suffix,"had minimal probability",new_suffix_relevance,\
						"of being visited from one of the prefs, and was ignored",file=self.prints_path,flush=True)
					self.number_ignored_suffixes_in_last_expand += 1
		return True

	def last_token_weight(self,prefix):
		if len(prefix)==0:
			return 1
		return self.prefix_weight(prefix)/self.prefix_weight(prefix[:-1]) # use own functions for prefix weight because they have memory

	def process(self,prefix): # fails if there was an inconsistency. 
		if not prefix in self.P: # check worthiness for addition to P 
			row = self.prefix_to_nprow(prefix)
			if len(self.get_matching_ps(row))>0: # this row is not in P (so only here for closedness check), and indeed closed: wrap it up
				return True 
			if self.last_token_weight(prefix)<self.interesting_p_transition_threshold: # this row isnt closed, but we dont care for it anyway
				self.skipped_P_count += 1
				if self.skipped_P_count%1e4==0:
					print("not expanding prefix:",prefix,"(last token weight is:",clean_val(self.last_token_weight(prefix),6),\
						"), have ignored:",self.skipped_P_count,"prefixes so far",file=self.prints_path,flush=True)
				return True
			# print("pref was not yet in P, has no matching rows, and is from a strong transition, so adding (and adding children to queue)"
			self.P.append(prefix) # unclosed, and not from worthless transition
			self.prefix_tree.insert(prefix,row) # P-prefs go in the prefix tree to be used and found in the future.
			# only ever add to the tree once. all those in the initial P are added on the expansion initiation.
			# new additions to P (only happens here) are processed here
			if len(self.P)%self.progress_P_print_rate == 0:
				print("|P|=",len(self.P),", time since extraction start:",clean_val(process_time()-self.table_start),file=self.prints_path,flush=True)
			# print("added pref to P")


		# (might occasionally get things that have already been accepted into P,\
		# eg through cexs. then their children have to be processed ('closedness') regardless), so we're out of the if now
		[self.queue_prefix(prefix+(t,)) for t in self.target.input_alphabet] 
		if len(self.S) >= self.max_S or len(self.P)>= self.max_P: 
			# time to stoppe, no point in adding more S's, i.e. checking consistency 
			# (if too many Ps then also no point adding more Ss, but if too many Ss return success and just stop checking 
			# consistency, might add a few more Ps for a while)
			return True
		return self.check_consistency(prefix)

	def queue_prefix(self,prefix):
		if prefix in self.been_queued: # been_queued is empty for every new expansion
			return # already got this one in thanks. will happen often once p has several entries, eg a aa aab, as children of some go into what P already has eg aa will try to add aab
		prefix_weight = self.prefix_weight(prefix)
		heappush(self.prefix_queue,(-prefix_weight,prefix))
		self.been_queued.add(prefix)

	def prefix_weight(self,prefix):
		res = self.prefix_weights_dict.get(prefix,None)
		if None is res:
			res = self.target.weight(prefix,as_prefix=True)
			self.prefix_weights_dict[prefix] = res
		return res

	def init_expansion(self):
		self.prefix_queue = []
		self.prefix_tree = KDTree(self.atol,interval_width = self.interval_width)
		self.been_queued = set() # to avoid double queueing and inserting things... may happen as P is prefix closed, and want to queue all extensions of each item in P, as well as all P...
		[self.queue_prefix(p) for p in self.P] # might be adding some it isn't interested in to get to the counterexample added to P, so have to use force
		
		[self.prefix_tree.insert(p,self.prefix_to_nprow(p)) for p in self.P]

	def expand(self):
		self.number_ignored_suffixes_in_last_expand = 0
		restart = True
		while restart:
			restart = False
			self.init_expansion()
			print("beginning expansion: |P|:",len(self.P),"|S|:",len(self.S),flush=True,file=self.prints_path)
			while self.prefix_queue and len(self.P)<self.max_P: 
				if (process_time() - self.table_start) > self.expanding_time_limit:
					print("reached max time, wrapping up",file=self.prints_path)
					break # have to start wrapping it up
				neg_w,prefix = heappop(self.prefix_queue)
				process_success = self.process(prefix)
				if not process_success: # something was added to S
					restart = True
					break # reinit the expansion
		print("finished expanding, |P|:",len(self.P),"|S|:",len(self.S),flush=True,file=self.prints_path)


	def add_counterexample(self,cex):
		print("adding counterexample:",cex)
		cex = tuple(cex) # just in case
		start_P = len(self.P)
		for n in range(len(cex)+1):
			if not cex[:n] in self.P:
				self.P.append(cex[:n])
		if not len(self.P) > start_P:
			print("cex did not add anything to P - it was all already here?",file=self.prints_path)
			print("cex was:",tup2seq(cex),file=self.prints_path)
			raise OhHeck()


class Relations:
	def __init__(self,table):
		self.table = table
		self.p2int = {p:i for i,p in enumerate(table.P)}
		n = len(table.P)
		self.mat = lil_matrix((n,n))
		self.epsilon = 1
		for i,p in enumerate(table.P):
			matches = table.get_matching_ps(table.prefix_to_nprow(p))
			match_indices = sorted([self.p2int[p] for p in matches])
			for j in match_indices:
				self.mat[i,j] = self.epsilon

	def match(self,p1,p2):
		i = self.p2int.get(p1,None)
		j = self.p2int.get(p2,None)
		if None in [i,j]:
			return self.table.equal(self.table.prefix_to_nprow(p1),self.table.prefix_to_nprow(p2))
		return self.mat[i,j] == self.epsilon

	def prefs_accept_pref(self,prefs,pref,enough_that_one_accepts=False):
		for p in prefs:
			if self.match(pref,p):
				if enough_that_one_accepts:
					return True
			else:
				return False
		return True

	def find_not_matching(self,prefs):
		return next(((p1,p2) for p1,p2 in combinations(prefs,2) if not self.match(p1,p2)),None)

	def is_clique(self,prefs):
		return None is self.find_not_matching(prefs)




class Minimiser:
	def __init__(self,table,prints_path):
		self.prints_path = prints_path
		self.table = table
		self.relations = Relations(table)
		self.input_alphabet = table.target.input_alphabet

	def set_cluster(self,p,c): # and its on you to clear the old c2ps to avoid duplicates before starting this
		self.p2c[p]=c
		self.c2ps[c].append(p)

	def clear_clusters(self,cs):
		for c in cs:
			self.c2ps[c]=[]

	def split_cluster_in_given_way(self,cluster,groups):
		max_cluster = max(list(self.c2ps.keys()))
		new_clusters =  [cluster] + [max_cluster+i for i in range(1,len(groups))] # might as well reuse c
		self.clear_clusters(new_clusters)
		for g,c in zip(groups,new_clusters):
			[self.set_cluster(p,c) for p in g]
		return new_clusters

	def make_clusters(self):
		do_timed(process_time(),self.init_clustering(),"cluster initialisation (dbscan)",file=self.prints_path)
		self.c2ps = inverse_dict(self.p2c,keys=self.table.P)
		print("dbscan made this many clusters               :",len(self.c2ps),",from",len(self.table.P),"prefixes",file=self.prints_path)
		do_timed(process_time(),self.refine_to_simulation(),"first refine to simulation",file=self.prints_path)
		print("refining to a simulation grew that to        :",len(self.c2ps),"clusters",file=self.prints_path)
		if do_timed(process_time(),self.refine_to_cliques(),"refining to cliques",file=self.prints_path): 
			print("refining to cliques made it:                 :",len(self.c2ps),file=self.prints_path)
			do_timed(process_time(),self.refine_to_simulation(),"second refine to simulation",file=self.prints_path)  # if cliques changed something, will need to look again
			print("and then refining to a simulation again made :",len(self.c2ps),file=self.prints_path)
			# no need to refine to cliques again after this, it was already cliques and the refining only splits, never merges
		else:
			print("refining to cliques made no more clusters",file=self.prints_path)


	def run(self):
		do_timed(process_time(),self.make_clusters(),"clustering the prefixes to clique+simulation requirements",file=self.prints_path)
		return do_timed(process_time(),self._to_pdfa(),"making pdfa from the clusters, ie mapping transitions etc",file=self.prints_path)

	def init_clustering(self):
		dbscan = DBSCAN(eps=self.relations.epsilon, metric='precomputed', min_samples=1)
		clusters = dbscan.fit(self.relations.mat).labels_.tolist()
		self.p2c = {p:c for p,c in zip(self.table.P,clusters)}


	def refine_to_simulation(self):
		while self.simulation_refinement():
			pass

	def simulation_refinement(self):
		for c in self.c2ps:
			P_prefs = self.c2ps[c]
			for t in self.input_alphabet:
				next_cs = set([self.p2c.get(p+(t,),None) for p in P_prefs])
				next_cs.discard(None) 
				if len(next_cs) > 1: # hecccc
					self.split_by_children(c,t,next_cs)
					return True
				# if len(next_cs) == 0: theyll just have to be sent to the best match somewhere later, in making the pdfa
				# finally, if len(next_cs) == 1, then thats perfect
		return False

	def split_by_children(self,cluster,token,children):
		child2g = {c:i for i,c in enumerate(children)}
		new_groups = [[] for _ in children]
		P_prefs = self.c2ps[cluster]
		unassigned_ps = []
		for p in P_prefs:
			child = self.p2c.get(p+(token,),None)
			if not None is child:
				new_groups[child2g[child]].append(p)
			else:
				unassigned_ps.append(p)
		new_clusters = self.split_cluster_in_given_way(cluster,new_groups)
		def just_best_cluster(p):
			r,_ = self.choose_best_cluster(p,new_clusters)
			return r
		[self.set_cluster(p,just_best_cluster(p)) for p in unassigned_ps]

	def clusters_with_a_chance(self,pref):
		row = self.table.prefix_to_nprow(pref)
		ps_with_a_chance = self.table.get_matching_ps(row) # all ps that are within tolerance of pref
		return list(set([self.p2c[p] for p in ps_with_a_chance])) # all clusters that have at least one p within tolerance of pref

	def choose_best_cluster(self,pref,all_options):
		def filtered_options():
			partial_match_options = [c for c in self.clusters_with_a_chance(pref) if c in all_options]
			clique_options = [c for c in partial_match_options if self.relations.prefs_accept_pref(self.c2ps[c],pref)]
			if clique_options:
				return clique_options, full_match_str
			if partial_match_options: # (even this might be empty bc there might be several unassigned p's, with the one that connected this pref to the original group being still unassigned)
				# filter to the ones that already aren't cliques, so p doesn't necessarily force additional splits
				options = [c for c in partial_match_options if not self.relations.is_clique(self.c2ps[c])]
				return (options if options else partial_match_options), partial_match_str
			return all_options, nothing_str
		def best_match(options):
			return options[0] # TODO: think of something better later
		options, match_level = filtered_options()
		return best_match(options), match_level
	
	def refine_to_cliques(self):
		all_clusters = list(self.c2ps.keys())
		something_happened = False
		for c in all_clusters: # the splitting into cliques will add new clusters to the end but not mess up stuff in the middle
			new_groups = self.split_into_cliques(self.c2ps[c])
			if len(new_groups)>1:
				self.split_cluster_in_given_way(c,new_groups)
				something_happened = True
		return something_happened

	def split_into_cliques(self,prefs): 
		if self.relations.is_clique(prefs):
			return [prefs]
		# TODO: in the recursive calls, it is possible DBSCAN will be able to split the observed prefs into cliques
		# best to try that first because it wont be too aggressive at any rate, whereas dimension-based splitting might 
		# get some stuff on the "wrong side" (maybe? idk but it definitely wont hurt to let dbscan have the first crack)
		rows = [self.table.prefix_to_nprow(p) for p in prefs]
		mins = [min([r[i] for r in rows]) for i in range(len(rows[0]))]
		maxs = [max([r[i] for r in rows]) for i in range(len(rows[0]))]
		ranges = [ma-mi for ma,mi in zip(maxs,mins)]
		for i in range(len(rows[0])):
			assert ranges[i] >= 0
		split_dim = np.argmax(ranges)
		if not ranges[split_dim] > self.table.atol: # wtf # otherwise this should be a clique?? (and so should already have returned)
			print("splitting up non-clique. biggest range is:",ranges[split_dim],"in dim:",split_dim,", atol is:",self.table.atol,file=self.prints_path)
			print("non-clique is:",prefs,file=self.prints_path)
			print("maxs are:",[clean_val(v) for v in maxs],file=self.prints_path)
			print("mins are:",[clean_val(v) for v in mins],file=self.prints_path)
			print("range is:",[clean_val(v) for v in ranges],file=self.prints_path)
			p1,p2 = self.relations.find_not_matching(prefs)
			print("unmatching prefs are: [",tup2seq(p1),"], [",tup2seq(p2),"]",file=self.prints_path)
			print("row for first is:",self.table.prefix_to_nprow(p1),file=self.prints_path)
			print("row for second is:",self.table.prefix_to_nprow(p2),file=self.prints_path)
			print("diff is:",self.table.prefix_to_nprow(p1)-self.table.prefix_to_nprow(p2),file=self.prints_path)
			raise OhHeck() # just to get the learner back innit
		max_r_over_atol = ranges[split_dim]/self.table.atol
		num_groups = int(np.ceil(max_r_over_atol))
		if num_groups == int(np.floor(max_r_over_atol)): # could happen, and will lead to missing group
			num_groups += 1
		new_groups = [[] for _ in range(num_groups)]
		for p,r in zip(prefs,rows):
			new_index = int(np.floor((r[split_dim]-mins[split_dim])/self.table.atol))
			new_groups[new_index].append(p)
		res = []
		for g in new_groups:
			res += self.split_into_cliques(g) # might end up with some recursion here
		return [r for r in res if r] # careful not to make empty cliques!!!

	def _to_pdfa(self):
		all_clusters = list(self.c2ps.keys())
		for c in all_clusters:
			assert self.c2ps[c] # make sure there arent any empty clusters
		transitions = {c:{} for c in all_clusters}
		transition_weights = {}
		forced = {}
		seen_states = []
		new_states = [self.p2c[()]]
		time_on_nonp_transitions = 0
		time_on_forcing_transitions = 0
		time_on_completely_bad_transitions = 0
		number_nonp_transitions = 0
		number_forced_transitions = 0
		number_completely_bad_transitions = 0
		for c in all_clusters:
			forced[c]={t:False for t in self.input_alphabet}
			for t in self.input_alphabet:
				p = next((p for p in self.c2ps[c] if p+(t,) in self.p2c),None)
				if None is p:
					all_ps = self.c2ps[c]
					if not all_ps:
						print("no prefixes for cluster:",c,file=self.prints_path)
						raise OhHeck()
					p = all_ps[np.argmax([self.table.prefix_weight(p) for p in all_ps])]
					start_transition_time = process_time()
					next_c, match_level = self.choose_best_cluster(p+(t,),all_clusters)
					making_transition_time = process_time() - start_transition_time
					time_on_nonp_transitions += making_transition_time
					number_nonp_transitions += 1
					if not match_level == full_match_str:
						time_on_forcing_transitions += making_transition_time
						number_forced_transitions += 1						
						forced[c][t] = True
					if match_level == nothing_str:
						time_on_completely_bad_transitions += making_transition_time
						number_completely_bad_transitions += 1
					# none of the pts are in P so we're not making any promises about them, but we'll do our best all the same
				else:
					next_c = self.p2c[p+(t,)]
				transitions[c][t] = next_c
		t2int = {t:self.table.S.index((t,)) for t in self.input_alphabet}
		for c in transitions:
			transition_weights[c]={}
			prefs = self.c2ps[c]
			rows = [self.table.prefix_to_nprow(p).tolist() for p in prefs]
			for t in self.input_alphabet:
				transition_weights[c][t] = np.mean([r[t2int[t]] for r in rows]) # TODO: make this weighted
		res = PDFA(transitions_and_weights=(transitions,transition_weights),
			end_token=self.table.target.end_token,initial_state=self.p2c[()])
		res.forced = forced # keep this info too
		print("total time spent on non-p transitions:",time_on_nonp_transitions,end=" ",file=self.prints_path)
		print("(non-p transitions:",number_nonp_transitions,", out of",len(all_clusters)*len(self.input_alphabet),"overall transitions)",file=self.prints_path)
		print("total time spent on forcing transitions:",time_on_forcing_transitions,end=" ",file=self.prints_path)
		print("(forced",number_forced_transitions,"transitions out of",len(all_clusters)*len(self.input_alphabet),")",file=self.prints_path)
		print("total time spent on completely bad transitions:",time_on_completely_bad_transitions,end=" ",file=self.prints_path)
		print("(completely randomly forced",number_completely_bad_transitions,"transitions out of",len(all_clusters)*len(self.input_alphabet),")",file=self.prints_path)

		return res


class CounterexampleGenerator:
	def __init__(self,target,atol,n_cex_attempts,max_counterexample_length,prints_path):
		self.prints_path = prints_path
		self.target = target
		self.atol = atol
		self.n_cex_attempts = n_cex_attempts
		self.max_counterexample_length = max_counterexample_length
		self.last_tokens = [(t,) for t in target.internal_alphabet]
		self.target_dict = {}

	def _get_target_row(self,pref):
		row = self.target_dict.get(pref,None)
		if None is row:
			row = np.array(self.target.last_token_probabilities_after_pref(pref,self.last_tokens))
			self.target_dict[pref] = row
		return row

	def _pref_to_rows(self,prefix):
		return self._get_target_row(prefix),\
		self.hypothesis.last_token_probabilities_after_pref(prefix,self.last_tokens) 
		# targets probably been checked on this val several times, 
		# hypothesis is new every time (and we don't recheck prefs on the same hypothesis)


	def _find_disagreeing_pref(self,w):
		w = tuple(w) # just in case
		for n in range(len(w)+1):
			pref = w[:n]
			if pref in self.checked:
				continue
			r1,r2 = self._pref_to_rows(pref)
			if not np.allclose(r1,r2,atol=self.atol):
				print("found disagreeing pref:",tup2seq(w[:n]),file=self.prints_path)
				print("hypothesis distribution after pref:",[clean_val(v) for v in r1],file=self.prints_path)
				print("target distribution after pref    :",[clean_val(v) for v in r2],file=self.prints_path)
				return pref
			self.checked.add(pref)
		return None

	def generate(self,hypothesis):
		assert set(hypothesis.internal_alphabet) == set(self.target.internal_alphabet) and (hypothesis.end_token == self.target.end_token) # order doesn't matter but they should have same letters and same EOS
		self.hypothesis = LanguageModel(hypothesis)
		self.checked = set()
		for n in range(self.n_cex_attempts):
			print("sample number:",n,"of",self.n_cex_attempts,file=self.prints_path)
			model = self.hypothesis if n%2 == 1 else self.target
			w = model.sample(cutoff=self.max_counterexample_length,empty_sequence=())
			pref = self._find_disagreeing_pref(w)
			if not None is pref:
				print("found cex on attempt",n,"of",self.n_cex_attempts,file=self.prints_path)
				print("found by sampling:",("hypothesis" if n%2==1 else "target"),file=self.prints_path)
				return pref
		print("no counterexamples found",file=self.prints_path)
		return None


def learn(target,max_P=np.inf,max_S=np.inf,max_states=np.inf,\
	pdfas_path=None,prints_path=None,atol=0.1,interval_width=0.2,\
	n_cex_attempts=1000,max_counterexample_length=1000,max_size=-1,\
	s_separating_threshold=-1,expanding_time_limit=np.inf,\
	progress_P_print_rate=1e4,interesting_p_transition_threshold=-1,very_verbose=False):
	prints_path = sys.stdout if None is prints_path else prints_path
	start = process_time()
	target = LanguageModel(target)
	table = Table(target,max_P,max_S,atol,interval_width,prints_path,s_separating_threshold,\
		expanding_time_limit,progress_P_print_rate,interesting_p_transition_threshold,very_verbose)
	cex_generator = CounterexampleGenerator(target,atol,n_cex_attempts,max_counterexample_length,prints_path)
	table.counterexamples = []
	table.extracted_sizes = []
	hypothesis = None
	interrupted = False
	hyp_counter = 1
	try:
		while True:
			obs_table_size = len(table.P)
			print("~~~~~~~~~~ starting expansion ~~~~~~~~~~~~~~",file=prints_path)
			do_timed(process_time(), table.expand(), "expanding observation table from size P="+str(obs_table_size),file = prints_path)
			print("got table with |P|:",len(table.P),", |S|:",len(table.S),file=prints_path)
			print("table has P (printing only 'edge' prefs):",file=prints_path)
			print_set_seqs(uniqPS(table.P,lambda x,y:x+(y,),target.input_alphabet),prints_path,join="\n*****\n")
			if very_verbose:
				print("\nP in full:",file=prints_path)
				print_set_seqs(table.P,prints_path,join="\n*****\n")
			print("\n\nand S (printing only 'edge' suffs):",file=prints_path)
			print_set_seqs(uniqPS(table.S,lambda x,y:(y,)+x,target.input_alphabet),prints_path,join=" §§ ")
			if very_verbose:
				print("\nS in full:",file=prints_path)
				print_set_seqs(table.S,prints_path,join=" §§ ")
			print("\n\nnow minimising",file=prints_path)
			minimiser = do_timed(process_time(),  Minimiser(table,prints_path), "making minimiser class for table (prepping relations for clustering)",file = prints_path)
			hypothesis = do_timed(process_time(), minimiser.run(), "making pdfa from table (after already prepping relations)",file = prints_path)
			print("\n\ngot hypothesis #",hyp_counter,", size:",hypothesis.n,file=prints_path)
			hypothesis.draw_nicely(max_size=max_size,filename=pdfas_path+"/"+str(hyp_counter),keep=True)
			hyp_counter += 1
			table.extracted_sizes.append(hypothesis.n)
			if len(table.P) >= max_P or len(table.S) >= max_S or hypothesis.n >= max_states or ((process_time()-table.table_start)>expanding_time_limit):
				break
			cex = do_timed(process_time(),cex_generator.generate(hypothesis), "searching for counterexample",file=prints_path)
			if None is cex:
				break
			table.counterexamples.append(cex)
			table.add_counterexample(cex)
	except (OhHeck,KeyboardInterrupt) as e:
		print("got exception:",e,file=prints_path)
		print("trace:",file=prints_path)
		traceback.print_exc(file=sys.stdout)
		interrupted = True

	creation_info = {"atol":atol,"interval width":interval_width,
					  "counterexamples":table.counterexamples,
					  "extracted sizes":table.extracted_sizes,
					  "final |S|":len(table.S),"final |P|":len(table.P),
					  "final S":table.S,"final P":table.P,
					  "max |S|":max_S,"max |P|":max_P,"max states":max_states,
					  "n_cex_attempts":n_cex_attempts,
					  "max_counterexample_length":max_counterexample_length,
					  "extraction time":process_time()-start,
					  "separating suffix threshold":s_separating_threshold,
					  "interesting p transition threshold":interesting_p_transition_threshold,
					  "number of ignored separating suffixes in last expand":table.number_ignored_suffixes_in_last_expand,
					  "expansion time limit":expanding_time_limit,
					  "num skipped prefixes":table.skipped_P_count}
	hypothesis.creation_info = creation_info				
	overwrite_file(hypothesis,pdfas_path+"/pdfa")			

	dict_str = lambda d,y:("\n".join([n+":"+str(d[n]) for n in d if not n in y]))
	print("\n\ncreation stats:\n",dict_str(creation_info,["final P","final S"]),file=prints_path,flush=True)

	# # for neatness' sake:
	# print("\n======\n final P (only edge of prefs, i.e. no 'aa' if has 'aab'):\n======\n",file=prints_path,flush=True)
	# uniqP = uniqPS(table.P,lambda x,y:x+(y,),target.input_alphabet)
	# print_set_seqs(uniqP,prints_path)
	# print("\n======\n final S (only edge of suffs, i.e. no 'aa' if has 'baa'):\n======\n",file=prints_path,flush=True)
	# uniqS = uniqPS(table.S,lambda x,y:(y,)+x,target.input_alphabet)
	# print_set_seqs(uniqS,prints_path)
	# print("\n".join(   [   ''.join(  [str(t) for t in s]  )      for s in uniqS]    ),file=prints_path,flush=True)

	return hypothesis, table, minimiser

def print_set_seqs(seqs,prints_path,join="\n"):
	print(join.join(   [   ''.join(  [str(t) for t in p]  )      for p in seqs]    ),file=prints_path,flush=True)

def uniqPS(PS,make_cont,input_alphabet):
	setPS = set(PS)
	def has_cont(p):
		conts = {make_cont(p,t) for t in input_alphabet}.intersection(setPS)
		return len(conts)>0
	return [p for p in PS if not has_cont(p)]


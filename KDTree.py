from math import ceil, floor

class KDNode:
	def __init__(self,tree):
		self.tree = tree
		self.elements = []
		self.children = []

	def insert(self,element,key):
		if len(key)>0: # non empty key, might be nparray but might also just be list so have to be careful with their junk
			if not self.children: # no children made yet
				self.children = [KDNode(self.tree) for _ in range(self.tree.n_intervals)] # make children on by-demand basis else get very hecked with lots of anyways-impossible branches right from the start
				self.n = len(self.children)
			return self.children[self.tree.interval(key[0])].insert(element,key[1:])
		else:
			if not element in self.elements:
				self.elements.append(element)
				self.tree.n_elements += 1
				return True
			return False

	def get_all_close(self,key,tolerance,element_to_remove=None): # tolerance is a scalar. key should be as deep as this node goes, meaning in the beginning it should be as deep as the tree
		if not self.children:
			return self.elements
		tolerance = tolerance + 1e-6 # bullshit to avoid idiot python doing things like 0.7/0.1=0.6999999 and not finding stuff
		res = []
		top = self.tree.interval(key[0]+tolerance)
		bottom = self.tree.interval(key[0]-tolerance)
		for i in range(bottom,top+1):
			res += self.children[i].get_all_close(key[1:],tolerance)
		return res

# assumes will always get key of same length!
class KDTree:
	def __init__(self,tolerance,min_val=0,max_val=1,interval_width=None):
		self.n_elements = 0
		self.interval_width = interval_width if not None is interval_width else (1.5*tolerance)
		self.tolerance = tolerance
		self.min = min_val
		self.max = max_val
		self.n_intervals = ceil((max_val-min_val)/self.interval_width)
		self.head = KDNode(self)

	def interval(self,val):
		return max(0,min(self.n_intervals-1,floor((val-self.min)/self.interval_width))) # zero indexed. with max to last interval and min to first because numerical errors and extreme values happen and sucks to be you when they do

	def insert(self,element,key): # key should be a vector of values inside min/max val. 
		return self.head.insert(element,key)

	def get_all_close(self,key,tolerance=None):
		tolerance = tolerance if not None is tolerance else self.tolerance
		return self.head.get_all_close(key,tolerance)
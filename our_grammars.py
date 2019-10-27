from PDFA import PDFA

def assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,initial_state):
	for s in transition_weights:
		total_noted = sum([transition_weights[s][a] for a in alphabet]) 
		assert total_noted <=1 and total_noted >= 0 # the remainder will be filled by the stopping token, but can't go over
	return PDFA(informal_name=informal_name,transitions_and_weights=(transitions,transition_weights),initial_state=initial_state)


def wgy1(): # wgy2 in personal
# description: (for max_as = 3)
# just a counting pdfa. alphabet is a,b. normally it prefers as,
# but whenever length(w)%(2+3+4) = 1, 4, or 8, it prefers bs. 
# (i.e. preference loops over: a, b, aa, b, aaa, b) 
# (for other max_as: loop carries on up to max_as a's in a row before the last b)
	max_as=3
	informal_name = "wgy_1"
	transitions = {}
	transition_weights = {}

	alphabet = [0,1]
	a,b = alphabet
	for i in range(sum(range(2,max_as+2))):
		transitions[i]={a:i+1,b:i+1}
		transition_weights[i]={a:0.75,b:0.15} 
	transitions[i]={a:0,b:0} # last one needs to loop back
	j=0
	for i in range(1,max_as+1):
		j+=i # skip the as
		transition_weights[j]={a:0.15,b:0.75} # places where b is higher
		j+=1 # skip the b
	return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def wgy2(): # wgy3 in personal
	# description:
	# just a counting pdfa. preference loops over all tokens.
	num_letters=5
	informal_name = "wgy_2"
	transitions = {}
	transition_weights = {}
	alphabet =  list(range(num_letters))
	for i in range(num_letters):
		next_state = (i+1)%num_letters
		transitions[i] = {a:next_state for a in alphabet}
		transition_weights[i] = {a:0.5/(num_letters+0.5) for a in alphabet} # divide 0.5 between all letters and stopping probability
		transition_weights[i][alphabet[i]] += 0.5 # give another 0.5 to this state's 'main' letter
	return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def wgy3(): # wgy7 in personal
	# description:
	# like tomita 5, but accept/reject is replaced with mild preference for a's or b's: 
	# whenever both #a(w) and #b(w) are odd, b is more likely, the rest of the time a is more likely
	hi=0.525
	lo=0.425
	informal_name = "wgy_3"
	transitions = {}
	transition_weights = {}
	alphabet = [0,1]
	a,b = alphabet
	transitions[0] = {a:1,b:2}
	transitions[1] = {a:0,b:3}
	transitions[2] = {a:3,b:0}
	transitions[3] = {a:2,b:1}
	for i in range(3):
		transition_weights[i] = {a:hi,b:lo}
	transition_weights[3] = {a:lo,b:hi}
	return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

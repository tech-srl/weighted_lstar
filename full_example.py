# TODO: move to process_time or time or something b/c clock is deprecated

from our_grammars import wgy1, wgy2, wgy3
from Helper_Functions import prepare_directory, overwrite_file, clean_val
from LanguageModel import LanguageModel
from Learner import learn
from NGram import NGram
import argparse
from spice_ndcg import modified_score_rankings
from RNNTokenPredictor import RNNTokenPredictor, train_rnn
from time import process_time
import math
from SpectralReconstruction import spectral_reconstruct, make_P_S
import os
import ast
import numpy as np
# example: going to train an RNN on WGY1, extract from it with weighted lstar, spectral, and ngrams 
# (for alergia - you will have to download and use the flexfringe toolkit), and then evaluate WER 
# and NDCG for each against the RNN


parser = argparse.ArgumentParser()
# language (either spice or wgy, it wont do both)
parser.add_argument('--spice-example',action='store_true')
parser.add_argument('--wgy-num',type=int,default=-1)

# train params
parser.add_argument('--RNNClass',type=str,default="LSTM",choices=["LSTM","GRU"])
parser.add_argument('--hidden-dim',type=int,default=50)
parser.add_argument('--input-dim',type=int,default=10)
parser.add_argument('--num-layers', type=int,default=2)
parser.add_argument('--dropout',type=float,default=0.5)
parser.add_argument('--learning-rates',type=ast.literal_eval,default=[0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005, 0.0001, 5e-05])
parser.add_argument('--iterations-per-learning-rate',type=int,default=10)
parser.add_argument('--batch-size',type=int,default=100)
parser.add_argument('--total-generated-train-samples',type=int,default=5000,help="only relevant when its going to make samples for you, ie when using a wgy") 
parser.add_argument('--max-generated-train-sample-len',type=int,default=200,help="only relevant when its going to make samples for you, ie when using a wgy") 

# # spectral learning params
parser.add_argument('--k-ranges',type=ast.literal_eval,default='[(1,31),(40,101,10)]')
parser.add_argument('--nPS',type=int,default=500)
parser.add_argument('--spectral-max-sample-attempts',type=int,default=int(1e5))
parser.add_argument('--spectral-max-sample-length',type=int,default=100)

# # lstar extraction params
parser.add_argument('--t-tol',type=float,default=0.1) # referred to as atol elsewhere
parser.add_argument('--interval_width',type=float,default=0.2) # generally keep it about 2*t-tol, but pointless if that ends up being close to 1
parser.add_argument('--dump-every',type=int,default=None)
parser.add_argument('--num-cex-attempts',type=int,default=500)
parser.add_argument('--max-counterexample-length',type=int,default=50)
parser.add_argument('--max-P',type=int,default=1000)
parser.add_argument('--max-states',type=int,default=math.inf) # effectively matching max-P by default
parser.add_argument('--max-S',type=int,default=50)
parser.add_argument('--lstar-time-limit',type=int,default=math.inf)
parser.add_argument('--progress-P-print-rate',type=int,default=math.inf)
parser.add_argument('--lstar-p-threshold',type=float,default=-1)
parser.add_argument('--lstar-s-threshold',type=float,default=-1)

# # ngram params
parser.add_argument('--ngram-total-sample-length',type=int,default=5e6)
parser.add_argument('--ngram-max-sample-length',type=int,default=1000)
parser.add_argument('--ngram-ns',type=ast.literal_eval,default='[1,2,3,4,5,6]')

# # eval params
parser.add_argument('--ndcg-num-samples',type=int,default=1000)
parser.add_argument('--ndcg-max-len',type=int,default=200)
parser.add_argument('--ndcg-k',type=int,default=5,help="probably best to try to use ndcg-k <= alphabet size tbh")
parser.add_argument('--wer-num-samples',type=int,default=1000)
parser.add_argument('--wer-max-len',type=int,default=100)


parser.add_argument('--code-test',action='store_true')

args = parser.parse_args()

if not args.spice_example and None is args.wgy_num:
	print("pick a spice or wgy")
	exit()

if args.code_test:
	args.hidden_dim = 10
	args.input_dim = 5
	args.num_layers = 2
	args.learning_rates = [0.01]
	args.iterations_per_learning_rate = 1
	args.total_generated_train_samples = 500
	args.nPS = 20
	args.spectral_max_sample_attempts = 100
	args.num_cex_attempts = 20
	args.max_P = 50
	args.max_S = 20
	args.lstar_time_limit = 20
	args.ngram_total_sample_length = 1e3
	args.ndcg_num_samples = 100
	args.wer_num_samples = 100
	args.wer_max_len = 10

args.k_list = []
for t in args.k_ranges:
	args.k_list += list(range(*t))

wgy = {1:wgy1(),2:wgy2(),3:wgy3()}
folder = "results"
prepare_directory(folder)

def make_spice_style_train(lm,n_samples,max_len,filename):
	prepare_directory(filename,includes_filename=True)
	with open(filename,"w") as f:
		print(n_samples,len(lm.internal_alphabet),file=f)
		for _ in range(n_samples):
			s = lm.sample(cutoff=max_len)
			print(len(s),*s,file=f)

def read_spice_style_train_data(filename):
	print("loading from file:",filename)
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

if args.spice_example:
	train_filename = "example_spice_data/0.spice.train"
	all_samples, alphabet = read_spice_style_train_data(train_filename)
	informal_name = "spice_0"
	rnn_folder = folder + "/"+ informal_name+"_"+str(process_time())
	prepare_directory(rnn_folder)
else:
	target = wgy[args.wgy_num]
	lm = LanguageModel(target)
	informal_name = target.informal_name
	rnn_folder = folder + "/"+ informal_name + "_" + str(process_time())
	prepare_directory(rnn_folder)
	print("making samples for",informal_name,end=" ... ")
	train_filename = rnn_folder+"/target_samples.txt"
	make_spice_style_train(lm,args.total_generated_train_samples,args.max_generated_train_sample_len,train_filename)
	print("done")
	target.draw_nicely(keep=True,filename=rnn_folder+"/target_pdfa")


all_samples, alphabet = read_spice_style_train_data(train_filename)

if len(alphabet) < args.ndcg_k:
	print("warning - using ndcg-k",args.ndcg_k,"with |alphabet|=",len(alphabet),
		"nothing technically wrong, just probably not very interesting results (all probably very high ndcg)")

train_frac = 0.9
val_frac = 0.05
train_stop = int(train_frac*len(all_samples))
val_stop = train_stop + int(val_frac*len(all_samples))
train_set = all_samples[:train_stop]
validation_set = all_samples[train_stop:val_stop]
test_set = all_samples[val_stop:]
print("have train, test, val for",informal_name)


training_prints_filename = rnn_folder + "/training_prints.txt" # this is the same one train_rnn will write into later, so just agree with it

rnn = RNNTokenPredictor(alphabet,args.input_dim,args.hidden_dim,args.num_layers,\
	args.RNNClass,dropout=args.dropout)

with open(training_prints_filename,"w") as f:
	print("currently training with train and validation sets of size:",\
		len(train_set),"(total len:",sum([len(s) for s in train_set]),"),",\
		len(validation_set),"(total len:",sum([len(s) for s in validation_set]),"), respectively",file=f)

print("made rnn, and beginning to train. will print train prints and final losses in:",training_prints_filename,flush=True)
train_start_time = process_time()
rnn = train_rnn(rnn,train_set,validation_set,rnn_folder,
		iterations_per_learning_rate=args.iterations_per_learning_rate,
		learning_rates=args.learning_rates,
		batch_size=args.batch_size,
		check_improvement_every=math.ceil(len(train_set)/args.batch_size)) 
		# might not return same rnn object as original prolly cause im doing it not how pytorch wants

def lapse_str(lapse,digs=2):
	return str(clean_val(lapse,digs))+"s ( "+str(clean_val(lapse/(60*60),2))+" hours)"

def clock_str(clock_start):
	return lapse_str(process_time()-clock_start,0)

print("done training (took ",clock_str(train_start_time),"), checking last losses on train, test, val and keeping in train prints file",flush=True)	
loss_start = process_time()

with open(training_prints_filename,"a") as f:
	print("\n\ntotal training time according to python's time.clock:",clock_str(train_start_time),file=f)
	print("\n\nultimately reached:\nlosses:",file=f)
	rnn_final_losses = {}
	rnn_final_losses["train"] = rnn.detached_average_loss_on_group(train_set)
	print("train loss:        ",rnn_final_losses["train"],file=f,flush=True)
	rnn_final_losses["validation"] = rnn.detached_average_loss_on_group(validation_set)
	print("validation loss:   ",rnn_final_losses["validation"],file=f,flush=True)
	rnn_final_losses["test"] = rnn.detached_average_loss_on_group(test_set)
	print("test loss:         ",rnn_final_losses["test"],file=f,flush=True)

print("done getting losses, that took:",clock_str(loss_start),flush=True)

print("beginning extractions! all will be saved and printed in subdirectories in",rnn_folder)

def do_lstar():
	print("~~~running weighted lstar extraction~~~")
	lstar_folder = rnn_folder+"/lstar"
	prepare_directory(lstar_folder)
	lstar_start = process_time()
	lstar_prints_filename = lstar_folder + "/extraction_prints.txt"
	print("progress prints will be in:",lstar_prints_filename)
	with open(lstar_prints_filename,"w") as f:
		lstar_pdfa,table,minimiser = learn(rnn,
			max_states = args.max_states,
			max_P = args.max_P,
			max_S=args.max_S,
			pdfas_path = lstar_folder,
			prints_path = f,
			atol = args.t_tol, 
			interval_width=args.interval_width,
			n_cex_attempts=args.num_cex_attempts,
			max_counterexample_length=args.max_counterexample_length,
			expanding_time_limit=args.lstar_time_limit,\
			s_separating_threshold=args.lstar_s_threshold,\
			interesting_p_transition_threshold=args.lstar_p_threshold,\
			progress_P_print_rate=args.progress_P_print_rate) 
	lstar_pdfa.creation_info = {"extraction time":process_time()-lstar_start,"size":len(lstar_pdfa.transitions)}
	lstar_pdfa.creation_info.update(vars(args)) # get all the extraction hyperparams as well, though this will also catch other hyperparams like the ngrams and stuff..
	overwrite_file(lstar_pdfa,lstar_folder+"/pdfa") # will end up in .gz
	with open(lstar_folder+"/extraction_info.txt","w") as f:
		print(lstar_pdfa.creation_info,file=f)
	lstar_pdfa.draw_nicely(keep=True,filename=lstar_folder+"/pdfa") # will end up in .img
	print("finished lstar extraction, that took:",clock_str(lstar_start))
	return lstar_pdfa
lstar_pdfa = do_lstar()


def do_ngram():
	print("~~~running ngram extraction~~~")
	print("making samples",end=" ... ")
	sample_start = process_time()
	samples = []
	length = 0
	lmrnn = LanguageModel(rnn)
	while length<args.ngram_total_sample_length:
		s = lmrnn.sample(cutoff=args.ngram_max_sample_length)
		samples.append(s)
		length += (len(s)+1) # ending the sequence is also a sample
	ngrams = {}
	ngrams_folder = rnn_folder + "/ngram"
	prepare_directory(ngrams_folder)
	sample_time = process_time() - sample_start
	print("done, that took:",clock_str(sample_start))
	print("making the actual ngrams",end=" ... ")
	with open(ngrams_folder+"/samples.txt","w") as f:
		print(len(samples),len(rnn.internal_alphabet),file=f)
		for s in samples:
			print(len(s),*s,file=f)
	for n in args.ngram_ns:
		ngram_start = process_time()
		ngram = NGram(n,rnn.input_alphabet,samples)
		ngram.creation_info = {"extraction time":sample_time+process_time()-ngram_start,"size":len(ngram._state_probs_dist),"n":n,
			"total samples len (including EOS)":length,"num samples":len(samples),"samples cutoff len":args.ngram_max_sample_length}
		overwrite_file(ngram,ngrams_folder+"/"+str(n))
		ngrams[n]=ngram
	with open(ngrams_folder+"/creation_infos.txt","w") as f:
		print("ngrams made from",len(samples),"samples, of total length",length,"(including EOSs)",file=f)
		for n in ngrams:
			print("===",n,"===\n",ngrams[n].creation_info,"\n\n",file=f)
	print("done, that took overall",clock_str(sample_start))
	return ngrams
ngrams = do_ngram()

def do_spectral():
	print("~~~running spectral extraction~~~")
	spectral_folder = rnn_folder +"/spectral_"+str(args.nPS)
	prepare_directory(spectral_folder)
	P,S = make_P_S(rnn,args.nPS,args.nPS,hard_stop=True,max_attempts=args.spectral_max_sample_attempts,\
				   max_sample_length=args.spectral_max_sample_length) 
	with open(spectral_folder+"/samples.txt","w") as f:
		sample_start = process_time()
		print("making P,S with n_PS:",args.nPS,end="...")
		print("done, that took:",clock_str(sample_start))
		sampling_time = process_time() - sample_start

		print("P (",len(P),") :\n\n",file=f)
		for p in P:
			print(*p,file=f)
		print("S (",len(S),") :\n\n",file=f)
		for s in S:
			print(*s,file=f)
	with open(spectral_folder+"/spectral_prints.txt","w") as f:
		print("getting P,S took:",sampling_time,file=f)
		wfas, times_excl_sampling, hankel_time, svd_time, _ = spectral_reconstruct(rnn,P,S,args.k_list,print_file=f)
		print("making hankels took:",hankel_time,file=f)
		print("running svd took:",svd_time,file=f)
		generic_creation_info = {"|P|":len(P),"|S|":len(S),"rnn name":rnn.name,
						 "hankel time":hankel_time,"svd time":svd_time}# ,"k":wfa.n} # "extraction time":total_time+PStime
		for wfa,t in zip(wfas,times_excl_sampling):
			wfa.creation_info = generic_creation_info
			wfa.creation_info["k"] = wfa.n
			wfa.creation_info["extraction time"] = t + sampling_time
			print("\n\n",wfa.n,"\n\n",wfa.creation_info,file=f)
			overwrite_file(wfa,spectral_folder+"/"+str(wfa.n))
	print("done, that took overall",clock_str(sample_start))
	return wfas
wfas = do_spectral()




# make ndcg file
# make wer file
def get_wer_samples():
	def all_prefs(test_set):
		res = set()
		for p in test_set:
			p = tuple(p) # make hashable, bit wonky to work like this but anyways will be consistent with LanguageModel expectations
			res.update(p[:i] for i in range(len(p)+1))
		return list(res)
	lm = LanguageModel(rnn)
	samples = [lm.sample(cutoff=args.wer_max_len) for _ in range(args.wer_num_samples)]
	gold_dict = lm.next_token_preds(all_prefs(samples))
	return samples, gold_dict


def get_ndcg_samples_and_target():
	lm = LanguageModel(rnn)
	prefs = []
	while len(prefs)< args.ndcg_num_samples:
		s = lm.sample(cutoff=args.ndcg_max_len)
		prefs += [s[:i] for i in range(len(s)+1)]
	prefs = prefs[:args.ndcg_num_samples] # remove extra ones possibly added by last sequence, just in name of easy reporting honestly
	prefs = list(prefs)
	with open(rnn_folder+"/ndcg_samples.txt","w") as f:
		print(len(prefs),len(lm.input_alphabet),file=f)
		for p in prefs:
			print(len(p)," ".join([str(t) for t in p]),file=f) # this is fine for the spices and for the wgys, where the tokens are ints. make sure to read it right too!
	target_filename = rnn_folder + "/ndcg_target.txt"
	with open(target_filename,"w") as f:
		print(args.ndcg_k,file=f) # store what ndcg_k is being made
		for p in prefs:
			d = lm.distribution_from_sequence(p)
			chars = sorted(list(d.keys()),key=lambda x:d[x],reverse=True)
			optimal = np.sum([d[c]/np.log2(i+2) for i,c in enumerate(chars[:args.ndcg_k])]) 
			#log2(i+2): ndcg wants i+1 where i is token index, but also remember enumerate starts from zero
			chars_weights = [v for pair in [(c,d[c]) for c in chars] for v in pair]
			chars_weights = [(v if not v == lm.end_token else -1) for v in chars_weights] 
			# spice scoring expects "-1" for end-of-sequence character
			print(optimal," ".join([str(t) for t in chars_weights]),file=f)
	return prefs, target_filename




with open(rnn_folder+"/final_results.txt","w") as f:
	print("trained rnn and ran extractions with parameters:\n",vars(args),file=f)
	print("computing wer over",args.wer_num_samples,"samples and ndcg with k=",args.ndcg_k,"over",args.ndcg_num_samples,"samples",file=f)
	print("rnn trained on lang",informal_name,"reached losses:",rnn_final_losses,file=f)

	wer_samples, wer_gold = get_wer_samples()
	ndcg_samples, ndcg_target_filename = get_ndcg_samples_and_target()
	def print_metrics(name,model,metric):
		lm = LanguageModel(model)
		if metric == "NDCG":
			temporary_model_preds_file = lm.make_spice_preds(ndcg_samples)
			ndcg = modified_score_rankings(temporary_model_preds_file,ndcg_target_filename)
			os.remove(temporary_model_preds_file)
			print(name,"got ndcg against rnn:",clean_val(ndcg,5),file=f)
		if metric == "WER":
			wer = lm.WER(wer_samples,gold_dict=wer_gold)
			print(name,"got wer against rnn:",clean_val(wer,5),file=f)
		if metric == "TIME":
			print(name,"took:",lapse_str(model.creation_info["extraction time"],1),"s",file=f)

	for metric in ["TIME","WER","NDCG"]:
		print("\n\n~~~~~~~~~~~~",metric,"~~~~~~~~~~~~",file=f)
		print("\n\n===LSTAR===",file=f)
		print_metrics("lstar",lstar_pdfa,metric)

		print("\n\n===NGRAM===",file=f)
		for n in sorted(list(ngrams.keys())):
			print_metrics(str(n)+"-gram",ngrams[n],metric)

		print("\n\n===SPECTRAL===",file=f)
		for w in wfas:
			print_metrics("wfa with rank: "+str(w.n),w,metric)

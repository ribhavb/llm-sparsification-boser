# Sparsity Analysis Report 

## Intro 
We assess the sparsity structures of three different models: RoBERTa, GPT2, and BART. Scripts for the sparsification of these models at the given sparsities can be found in the create_sparse_models.py script, where we use a global unstructured pruning method (L1 Norm). We then evaluate these different models and their varying sparsities on two benchmark tasks within GLUE: MRPC, and SST2. 

SST2 -- or "Stanford Sentiment Treebank Classification" is a sentiment analysis task using excerpts from movie reviews as the data source. 

MRPC -- or "Microsoft Research Paraphrase Corpus" is a paraphrasing task where the model is given two sentences and asked whether one is a paraphrase of the other. 

(Details on these tasks and others in the GLUE benchmark can be found here: https://docs.google.com/spreadsheets/d/1BrOdjJgky7FfeiwC_VDURZuRPUFUAz_jfczPPT35P00/edit#gid=0)


## Task Results 

![sst2_accuracy_results][./src/images/sst2_accuracy.png]

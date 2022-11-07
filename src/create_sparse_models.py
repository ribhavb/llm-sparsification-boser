import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune
import numpy as np
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import T5Tokenizer, T5Model

def get_children(m): 
    child = [c for c in m.children()]
    if len(child) > 0: 
        res = []
        for c in child:
            for x in get_children(c):
                res.append(x)
        return res
    else:
        return [m]

def get_prune_layers_no_bias(model):
    to_prune = []
    layers = get_children(model)
    for l in layers:
        p = l.named_parameters()
        for obj in p:
            if obj[0] == 'weight' or obj[0] == 'weight_orig':
                to_prune.append((l,obj[0]))
                
    return to_prune
                

def get_prune_layers_with_bias(model):
    to_prune = []
    layers = get_children(model)
    for l in layers:
        p = l.named_parameters()
        for obj in p:
            to_prune.append((l,obj[0]))
            
    return to_prune

def full_model_prune(model, sparsity):
#     model.toggle_pruning(True)
    to_prune = get_prune_layers_no_bias(model)
    print(to_prune)
    
    prune.global_unstructured(
        to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,)
    
    print('finished prune')
    for (layer,wt) in to_prune:
        print('in remove')
        print(list(layer.named_buffers()))
        try:
            prune.remove(layer, wt)
        except Exception as e:
            print(e)

    
    return model

sparsity_values = [0.1,0.5,0.9,0.95,0.99]

for sparsity in sparsity_values:
    print(sparsity)
    model = GPT2Model.from_pretrained("gpt2")
    model.config.pad_token_id = model.config.eos_token_id
    model = full_model_prune(model,sparsity)   
    model.save_pretrained(f"gpt2models/gpt2_{sparsity}")

model = GPT2Model.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
model.save_pretrained(f"gpt2models/gpt2")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(f"gpt2models/gpt2_tokenizer")

for sparsity in sparsity_values:
    model = T5Model.from_pretrained("t5-small")
    model = full_model_prune(model,sparsity)
    model.save_pretrained(f"t5models/t5_{sparsity}")

model = T5Model.from_pretrained("t5-small")
model.save_pretrained(f"t5models/t5_small")

for sparsity in sparsity_values:
    print(sparsity)
    roberta_model = RobertaModel.from_pretrained("roberta-base")
    roberta_model = full_model_prune(roberta_model, sparsity)
    roberta_model.save_pretrained(f"roberta_models/roberta_{sparsity}")

roberta_model = RobertaModel.from_pretrained("roberta-base") 
roberta_model.save_pretrained(f"roberta_models/roberta_base")
from collections import defaultdict, deque
import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from gradiend.data import read_article_ds
from gradiend.util import hash_it
from torch.utils.data.sampler import Sampler

class DeTrainingDataset(Dataset):
    def __init__(self, data, config, tokenizer, batch_size, max_token_length=48, is_generative=False):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.config = config

        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token

        articles = self.config['articles']

        self.article_tokens = {(determiner, upper): self.tokenizer.encode(
            determiner[0].upper() + determiner[1:] if upper else determiner,
            add_special_tokens=False)[0] for determiner in articles for upper in [True, False]}


    """Returns the number of entries in the dataset"""
    def __len__(self):
        return len(self.data)


    '''Returns the tokenIds and the attention masks as a torch tensor.'''
    def tokenize(self, text):
        item = self.tokenizer(text, truncation=True, padding='max_length',
                              max_length=self.max_token_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in item.items()}
        return item

 
    def __getitem__(self, index):
        entry = self.data.iloc[index]
        is_gender_data=True
        
        if is_gender_data:
            masked_entry = entry['masked']
           
            key = entry['dataset_label']
            determiner = entry['label'].lower()
            inverse_determiner = self.config[key]['inverse']
         
            def fill(text):
                return text.replace(self.config[key]['mask'], self.mask_token)
               

            gender_text = fill(masked_entry)
    
            item = self.tokenize(gender_text)
            gender_labels = item['input_ids'].clone()
            inv_gender_labels = gender_labels.clone()

            sentence_delimiter = {'.', '!', '?'}
        
            mask_token_mask = gender_labels == self.mask_token_id

            gender_text_no_white_spaces = gender_text.replace(' ', '')

            if self.mask_token not in gender_text_no_white_spaces: 
                print("mask_index not found for",  entry['masked'])
            
            mask_index = gender_text_no_white_spaces.index(self.mask_token)
            upper = mask_index == 0 or mask_index > 2 and gender_text_no_white_spaces[
                mask_index - 1] in sentence_delimiter and gender_text_no_white_spaces[mask_index - 2] != '.'  

            # only compute loss on masked tokens
            gender_labels[~mask_token_mask] = -100
            gender_labels[mask_token_mask] = self.article_tokens[(
                determiner, upper)]  

            inv_gender_labels[~mask_token_mask] = -100
            inv_gender_labels[mask_token_mask] = self.article_tokens[(
                inverse_determiner, upper)] 

            inv_item = item.copy()
            item['labels'] = gender_labels
            inv_item['labels'] = inv_gender_labels

           
            label = self.config[key]['code'] 
            text = gender_text
        else:
            text = entry['text']

            item = self.tokenize(text)
            masked_input, labels = self.mask_tokens(item['input_ids'])
            item['input_ids'] = masked_input
            item['labels'] = labels
            inv_item = item
            label = ''

        return {True: item, False: inv_item, 'text': text, 'label': label, 'dataset_label': key}    


def create_de_training_dataset(tokenizer, config, max_size=None, batch_size=None, split=None, article=None, is_generative=False):

    dataset = read_article_ds(split=split, article=article)
    if max_size:
        dataset = dataset.iloc[range(max_size)]

    return DeTrainingDataset(dataset, config, tokenizer, batch_size=batch_size, is_generative=is_generative)


def create_de_eval_dataset(gradiend, config, max_size=None, split='val', source='gradient', save_layer_files=False, is_generative=False):
    if not source in {'gradient', 'inv_gradient', 'diff'}:
        raise ValueError(f'Invalid source {source}')

    start = time.time()

    eval_datasets = []
    for label in config['combinations']:
        eval_dataset = create_de_training_dataset(gradiend.tokenizer, config, article=label, split=split)
        eval_dataset.data = eval_dataset.data.head(120)
        eval_datasets.append(eval_dataset)

    dataset = CombinedDataset(eval_datasets)

    texts = dataset.data.loc[:, ['masked', 'label', 'dataset_label']]
    
    texts = dataset.data.sample(frac=1, random_state=42).reset_index(drop=True)
    #texts = shuffled.loc[:max_size, ['masked', 'label']]
    if max_size:
        if 0.0 <= max_size <= 1.0:
            max_size = int(max_size * len(texts))
            print('eval max_size', max_size)
        texts = texts[:(max_size)]

    mask_token = gradiend.tokenizer.mask_token

   
    filled_texts = {}
    for _, row in texts.iterrows():
        text = row['masked']
        key = row['dataset_label']
       
        filled_text = text.replace(config[key]['mask'], mask_token)
        filled_texts[filled_text] = {"label": row['label'], "dataset_label": row['dataset_label']}


    # calculate the gradients in advance, if not already cached?
    base_model = gradiend.base_model.name_or_path
    base_model = os.path.basename(base_model)
    layers_hash = gradiend.layers_hash
    cache_dir = f'data/cache/gradients/{base_model}/{source}/{layers_hash}'
    gradients = defaultdict(dict)  # maps texts to the gradients
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # the actual evalution data being loaded and the gradients being calculated
    text_iterator = tqdm(
        filled_texts, desc=f'Loading cached evaluation data', leave=False)
    layers_hash = hash_it(gradiend.gradiend.layers)
    for i, filled_text in enumerate(text_iterator):
        text_hash = hash_it(filled_text)

        def create_layer_file(layer):
            return f'{cache_dir}/{text_hash}/{layer}.pt'

        cached_tensor_file = f'{cache_dir}/tensor_{text_hash}.pt'
        if os.path.exists(cached_tensor_file):
            gradient = torch.load(cached_tensor_file).half().cpu()
            gradients[filled_text] = gradient
            continue

        # first check whether we need to calculate the gradients
        requires_grad = any(not os.path.exists(create_layer_file(layer))
                            for layer in gradiend.gradiend.layers)

        # only compute the gradients (computationally expensive) if really needed
        if requires_grad:
            print(f"Calculate gradients for {filled_text} and label {filled_texts[filled_text]['label']}")

            label = filled_texts[filled_text]['label'].lower()
            inv_label = config[filled_texts[filled_text]['dataset_label']]['inverse']

            if source == 'diff':
                label_factual = label
                label_counter_factual = inv_label

                inputs_factual = gradiend.create_inputs(
                    filled_text, label_factual)
                grads_factual = gradiend.forward_pass(
                    inputs_factual, return_dict=True)
                inputs_counter_factual = gradiend.create_inputs(
                    filled_text, label_counter_factual)
                grads_counter_factual = gradiend.forward_pass(
                    inputs_counter_factual, return_dict=True)
                grads = {layer: grads_factual[layer] - grads_counter_factual[layer]
                         for layer in gradiend.gradiend.layers}
            else:
                if source == 'gradient':
                    label = label
                elif source == 'inv_gradient':
                    label = inv_label
                    #config[filled_texts[filled_text]['dataset_label']]['inverse']

                inputs = gradiend.create_inputs(filled_text, label)
                grads = gradiend.forward_pass(inputs, return_dict=True)

            if save_layer_files:
                # create the directory
                dummy_file = create_layer_file('dummy')
                os.makedirs(os.path.dirname(dummy_file), exist_ok=True)
        else:
            grads = None

        for layer in gradiend.gradiend.layers:
            layer_file = create_layer_file(layer)
            if not os.path.exists(layer_file):
                weights = grads[layer].half().flatten().cpu()
                if save_layer_files:
                    # Saving individual layer files doubles the necessary storage, but is more efficient when working with different layer subsets
                    torch.save(weights, layer_file)
            else:
                weights = torch.load(layer_file, weights_only=False)

            # convert back to float32 for consistency with other parameters
            weights = weights.float()
            gradients[filled_text][layer] = weights

        # convert layer dict to single tensor
        full_gradient = torch.concat(
            [v for v in gradients[filled_text].values()], dim=0).half()
        gradient = full_gradient
        if isinstance(gradiend.gradiend.layers, dict):
            mask = torch.concat(
                [gradiend.gradiend.layers[k].flatten() for k in gradients[filled_text].keys()], dim=0
            ).cpu()
            gradient = full_gradient[mask]

        gradients[filled_text] = gradient
        torch.save(gradient, cached_tensor_file)


    labels = {k: config[v['dataset_label']]['code'] for k,v in filled_texts.items()}
    print(labels)
    result = {
        'gradients': gradients,
        'labels': labels
    }

    print(
        f'Loaded the evaluation data with {len(gradients)} entries in {time.time() - start:.2f}s')

    return result




class SingleLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
    
        self.labels = defaultdict(list)

        for idx in range(len(dataset)):
            label = dataset[idx]['dataset_label']
            self.labels[label].append(idx)

        self.batches = self._create_batches()
        print('#labels in the Sampler', len(self.labels))


    def _create_batches(self): 
        label_batches = {}
        for label, indices in self.labels.items():
            label_batches[label] = deque()
            for i in range(0, len(indices), self.batch_size): 
                batch = indices[i:i+ self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                label_batches[label].append(batch)

        # batches are interleaved, so it goes label 1, label 2, label 1 etc... 
        interleaved = []   
        label_cycle = list(label_batches.keys())


        while any(label_batches.values()):
            for label in label_cycle: 
                if label_batches[label]:
                    batch = label_batches[label].popleft()
                    interleaved.append(batch)

        return interleaved         
             

    def __iter__(self):
        for batch in self.batches: 
            yield batch
    

    def __len__(self): 
        return len(self.batches)                  
                    



class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.data = pd.concat([dataset.data for dataset in self.datasets], ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]  
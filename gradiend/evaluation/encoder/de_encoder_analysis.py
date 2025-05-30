import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch
from tqdm import tqdm
from gradiend.data.util import get_file_name, json_dumps
from gradiend.evaluation.encoder.encoder_analysis import EncoderAnalysis, get_pearson_correlation, get_spearman_correlation, z_score
from gradiend.util import get_files_and_folders_with_prefix



class DeEncoderAnalysis(EncoderAnalysis):
    def __init__(self, config):
        super().__init__(config)


    def analyse_encoder(self, model_with_gradiend, dataset, output, plot=False):
        model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        mask_token = tokenizer.mask_token 
    
        cache_default_predictions_dict = self.read_default_predictions(model)

        modified_cache = []
        
        def get_de_default_predictions(masked_text): 
            if masked_text in cache_default_predictions_dict: 
                return cache_default_predictions_dict[masked_text]
        
            predictions = self.evaluate_determiners(model, tokenizer, masked_text)
            cache_default_predictions_dict[masked_text] = predictions

            if not modified_cache: 
                modified_cache.append(True)
            return predictions 
    
        
        source = model_with_gradiend.gradiend.kwargs['training']['source']
   

        filled_texts = []
        default_preds = self.config['default_predictions']

        def process_entry(row, plot=False):
    
            key = row['dataset_label']        
            masked = row['masked'] 
            encoded_values = []
            articles = []
            labels = []
            dataset_labels=[]
            default_predictions = {k: []
                               for k in default_preds}
        
            inputs = []

            filled_text = masked.replace(self.config[key]['mask'], mask_token)  
            filled_texts.append(filled_text)  

            label = row['label'].lower()

            #TODO own function that returns the encoded value 
            if source == 'diff':
                label_factual = label
                label_counter_factual = self.config[key]['inverse']

                inputs_factual = model_with_gradiend.create_inputs(
                    filled_text, label_factual)
                grads_factual = model_with_gradiend.forward_pass(
                    inputs_factual, return_dict=False)
                inputs_counter_factual = model_with_gradiend.create_inputs(
                    filled_text, label_counter_factual)
                grads_counter_factual = model_with_gradiend.forward_pass(
                    inputs_counter_factual, return_dict=False)
                grads = grads_factual - grads_counter_factual
                inputs.append(grads)
                encoded = model_with_gradiend.gradiend.encoder(grads).item()
            else:
                if source == 'gradient':
                    masked_label = label
                elif source == 'inv_gradient':
                    masked_label = self.config[key]['inverse']
                else:
                    raise ValueError(f'Unknown source: {source}')
            
                inputs.append((filled_text, masked_label))
                encoded = model_with_gradiend.encode(filled_text, label=masked_label)

                encoded_values.append(encoded)
                articles.append(row['label'])
                dataset_labels.append(row['dataset_label'])
        
          
                labels.append([label] * row['token_count'])
           
                default_prediction = get_de_default_predictions(filled_text)
        
                default_prediction['label'] = label

                

                for key, value in default_prediction.items():
                    default_predictions[key].append(value)

                unique_labels = [list(item) for item in set(tuple(x) for x in labels)]

        
            
            results = pd.DataFrame({
                'text': masked,
                'state': articles,
                'dataset_labels': dataset_labels,
                'encoded': encoded_values,
                'labels': unique_labels,
                'type': f"{self.det_combination} masked",
                **default_predictions,
            	})

            results['state_value'] = results['dataset_labels'].map(lambda dataset_label: self.config[dataset_label]['code'])
            results['z_score'] = z_score(results['encoded'])
            results = results.sort_values(by='state')

            if plot:
                plt.title(row['masked'])
                sns.boxplot(x='state', y='z_score', data=results)
                plt.show()

 
            return results
    
        tqdm.pandas(desc=f"Analyze with {self.det_combination} Test Data")
        dataset[f"{self.det_combination}"] = dataset.progress_apply(process_entry, axis=1)
   
        results = dataset[f"{self.det_combination}"].tolist()

        texts = []
        encoded_values = []
        labels = []
        default_predictions = {k: []
                           for k in default_preds}
    
        tokens_to_ignore = set(self.config['token_to_ignore'])
        ingore_tokens= list(set((token for det in tokens_to_ignore for token in tokenizer(det, add_special_tokens=False )['input_ids'])))

        torch.manual_seed(42)
        #TODO this right now is not that important, i dont have the right dataset for this. 
        for text in tqdm(filled_texts, desc=f"{self.det_combination} data without determiners masked"):
            encoded, masked_text, label = model_with_gradiend.mask_and_encode(
                text, ignore_tokens=ingore_tokens, return_masked_text=True)
            texts.append(text)
            encoded_values.append(encoded)
            labels.append(label)
        

            default_prediction = get_de_default_predictions(masked_text)
            default_prediction['label'] = label
            for key, value in default_prediction.items():
                default_predictions[key].append(value)

          
        result = pd.DataFrame({
            'text': texts,
            'state': None,
            'dataset_labels': None,
            'encoded': encoded_values,
            'type': f"no {self.det_combination} masked",
            **default_predictions,
        })
        results.append(result)

 
    
        if modified_cache:
            self.write_default_predictions(cache_default_predictions_dict, model)

        total_results = pd.concat(results)

        mean = total_results['encoded'].mean()
        std = total_results['encoded'].std()
        total_results['global_z_score'] = (total_results['encoded'] - mean) / std

    
        for article in self.articles: 
            total_results[article] = total_results[article].apply(json_dumps)

   
        total_results['label'] = total_results['label'].apply(json_dumps)
        total_results['most_likely_token'] = total_results['most_likely_token'].apply(
            json_dumps)

        total_results.to_csv(output, index=False)

        if plot:
            # plot results
            self.plot_model_results(total_results, title=output.removesuffix('.csv'))

            plot_results = total_results[total_results['type'] == f"{self.det_combination} masked"].sort_values(
                by='state').reset_index(drop=True)
            sns.boxplot(x='state', y='encoded', data=plot_results)
            plt.title(model_with_gradiend.name_or_path)
            plt.show()

            cor = np.nanmean([text_df[['encoded', 'state_value']].corr(method='pearson')[
                         'encoded']['state_value'] for text, text_df in plot_results.groupby('text')])
            print('Correlation', cor)

        
            plot_results_MF = plot_results[plot_results['state'] != 'B']
            cor = np.nanmean([text_df[['encoded', 'state_value']].corr(method='pearson')[
                         'encoded']['state_value'] for text, text_df in plot_results_MF.groupby('text')])
            print('Correlation MF', cor)

        return total_results


    def get_model_metrics(self,*encoded_values, prefix=None, suffix='.csv', **kwargs):
        if prefix:
        # find all models in the folder with the suffix
            encoded_values = list(encoded_values) + get_files_and_folders_with_prefix(prefix, suffix=suffix)

        if len(encoded_values) > 1:
            metrics = {}
            for ev in encoded_values:
                m = self.get_model_metrics(ev, **kwargs)
                metrics[ev] = m

            return metrics

        raw_encoded_values = encoded_values[0]

        encoded_values = get_file_name(raw_encoded_values, file_format='csv', **kwargs)
        json_file = encoded_values.replace('.csv', '.json')

        try:
            return json.load(open(json_file, 'r+'))
        except FileNotFoundError:
            print('Computing model metrics for', encoded_values)

        df_all = pd.read_csv(encoded_values)
        try:
            df = df_all[df_all['type'] == f"{self.det_combination} masked"]
        except KeyError:
            df = df_all

        #df_without_B = df[df['state'] != 'B'].copy()


        df[f"z_score_{self.det_combination}"] = z_score(df, key='encoded', groupby='text')
        df[f"global_z_score_{self.det_combination}"] = z_score(df['encoded'])

        #TODO the acc_M_pos/new can be calculated using the dataset_labels, more robust... 
        df['state_value'] = df['state_value'].apply(lambda x: 1 if x in [0,1,2,3] else 0)

        acc_M_positive = np.mean([((text_df['encoded'] >= 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df.groupby('text')])
        acc_M_negative = np.mean([((text_df['encoded'] < 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df.groupby('text')])
    
        acc_optimized_border_M_pos = np.mean([max(((text_df['encoded'] < threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df.groupby('text')])
        acc_optimized_border_M_neg = np.mean([max(((text_df['encoded'] > threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df.groupby('text')])

        acc_M_positive_global = np.mean([((text_df[f"global_z_score_{self.det_combination}"] >= 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df.groupby('text')])
        acc_M_negative_global = np.mean([((text_df[f"global_z_score_{self.det_combination}"] < 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df.groupby('text')])

        acc_optimized_border_M_pos_global = np.mean([max(((text_df[f"global_z_score_{self.det_combination}"] < threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df.groupby('text')])
        acc_optimized_border_M_neg_global = np.mean([max(((text_df[f"global_z_score_{self.det_combination}"] > threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df.groupby('text')])


        # rename the keys such that blanks are replaced by '_'
        df_all = df_all.rename(columns=lambda x: x.replace(' ', '_'))
        encoded_abs_means = df_all.groupby('type')['encoded'].apply(lambda group: group.abs().mean()).to_dict()
        encoded_means = df_all.groupby('type')['encoded'].apply(lambda group: group.mean()).to_dict()
   
        # map encoded values to the predicted class, i.e. >= 0.5 -> female, <= -0.5 -> male, >-0.5 & <0.5 -> neutral
        
        df_all['predicted_female_pos'] = df_all['encoded'].apply(lambda x: 1 if x >= 0.5 else (-1 if x <= -0.5 else 0))
        df_all['predicted_male_pos'] = df_all['encoded'].apply(lambda x: 1 if x <= -0.5 else (-1 if x >= 0.5 else 0))

        gender_keys = list(self.config['categories'].keys())
    
        #TODO: there is a better way to do this, right now its very finnicky
        # labels = df_all['dataset_labels'].apply(lambda x: 1 if x in self.config['categories'][gender_keys[1]]['labels'] else (-1 if x in config['categories'][gender_keys[0]]['labels'] else 0))
        # df_labels = df['dataset_labels'].apply(lambda x: 1 if x in self.config['categories'][gender_keys[1]]['labels'] else (-1 if x in config['categories'][gender_keys[0]]['labels'] else 0))

        df_all_labels = df_all['dataset_labels'].apply(lambda x: self.config.get(x, {}).get('encoding', 0)).astype(int)
        df_labels = df['dataset_labels'].apply(lambda x: self.config.get(x, {}).get('encoding', 0)).astype(int)

        df_all['state_value'] = df_all_labels
        df['state_value'] = df_labels

        balanced_acc_female_pos = balanced_accuracy_score(df_all['predicted_female_pos'], df_all_labels)
        balanced_acc_male_pos = balanced_accuracy_score(df_all['predicted_male_pos'], df_all_labels)
        acc_total = max(balanced_acc_female_pos, balanced_acc_male_pos)
      

        pearson_total = get_pearson_correlation(df_all)
        spearman_total = get_spearman_correlation(df_all)

        pearson = get_pearson_correlation(df)
        spearman = get_spearman_correlation(df)


        scores = {
            'pearson_total': pearson_total['correlation'],
            'pearson_total_p_value': pearson_total['p_value'],
            'spearman_total': spearman_total['correlation'],
            'spearman_total_p_value': spearman_total['p_value'],
            'acc_total': acc_total,

            'pearson': pearson['correlation'],
            'pearson_p_value': pearson['p_value'],
            'spearmann': spearman,
            'spearman_p_value': spearman['p_value'],

            'acc': max(acc_M_negative, acc_M_positive),
            'acc_zscore': max(acc_M_negative_global, acc_M_positive_global),
            'acc_optimized': max(acc_optimized_border_M_neg, acc_optimized_border_M_pos),
            'acc_optimized_zscore': max(acc_optimized_border_M_neg_global, acc_optimized_border_M_pos_global),

            'encoded_abs_means': encoded_abs_means,
            'encoded_means': encoded_means,

            **self.get_std_stats(df),
        }

        print(scores)


        with open(json_file, 'w') as f:
            json.dump(scores, f, indent=4)

        return scores

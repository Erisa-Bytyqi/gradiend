import csv
import os
import pandas as pd
import torch
from torch import device, softmax

from gradiend.evaluation.decoder.decoder_analysis import DecoderAnalysis
from gradiend.evaluation.mlm import evaluate_clm_perplexity, evaluate_mlm
from gradiend.data import read_de_neutral

from itertools import chain
from typing import Tuple
import numpy as np
import torch
import os
import logging
import pandas as pd
from gradiend.model import ModelWithGradiend
from gradiend.model import ModelWithGradiend

log = logging.getLogger(__name__)


class DeDecoderAnalysis(DecoderAnalysis):

    def __init__(self, model, tokenizer):
        super().__init__()

        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def default_evaluation():
        pass

    def evaluate_decoder():
        # here i call the call single_gram for each lr and gender factor
        # two models, bert and distilbert
        # each call produces exactly ONE line of scores
        # so my row:
        # lr, factor, acc, acc_det, acc_adj, etc and then plot map...
        pass

    # TODO this doesnt process batches correctly
    def prepare_texts(self, sentence, method="salazar"):
        sentence_split = sentence.split(" ")
        encoded_new = self.tokenizer(
            sentence_split,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        token_idx = encoded_new["input_ids"]
        attentions_masks = encoded_new["attention_mask"]

        masked_tensors = []

        # original mask_indices >> [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        # tokens = encoded.tokens() >> ['[CLS]', 'the', 'ho', '##oli', '##gan', 'wrecked', 'the', 'vehicle', '.', '[SEP]', '[PAD]', '[PAD]']
        # word_idx = encoded.word_ids() >> [None, 0, 1, 1, 1, 2, 3, 4, 5, None, None, None]
        # we mask out tokens to the right of current token for multi-token words. Use word_idx list for this purpose
        # dapted mask_indices outputs >> [[0], [1], [2, 3, 4], [3, 4], [4], [5], [6], [7], [8], [9]]

        for index, (token_ids, attention_mask) in enumerate(
            zip(token_idx, attentions_masks)
        ):
            tokens = encoded_new["input_ids"][0]
            word_ids = encoded_new.word_ids()

            token_ids = tokens
            attention_mask = attention_mask.clone()

            token_ids_masked_list = []
            attention_masked_list = []

            # basically do not mask any cls or whatever token or include in the calculation

            effective_token_ids = [
                token_id
                for token_id in token_ids
                if token_id != self.pad_token_id
                and token_id != self.cls_token_id
                and token_id != self.sep_token_id
            ]
            effective_length = len(effective_token_ids)

            if method == "kauf_l2r":
                mask_indices = [
                    (
                        [mask_pos]
                        + [
                            j
                            for j in range(mask_pos + 1, effective_length + 2)
                            if word_ids[j] == word_ids[mask_pos]
                        ]
                        if word_ids[mask_pos] is not None
                        else [mask_pos]
                    )
                    for mask_pos in range(effective_length + 2)
                ]

            elif method == "salazar":
                mask_indices = [[mask_pos] for mask_pos in range(effective_length + 2)]

            else:
                raise NotImplementedError

            mask_indices = mask_indices[1:-1]

            for mask_set in mask_indices:
                token_ids_masked = token_ids.clone()
                token_ids_masked[mask_set] = self.mask_token_id

                attention_masked = attention_mask.clone()

                token_ids_masked_list.append(token_ids_masked)
                attention_masked_list.append(attention_masked)

            masked_tensors.append(
                (
                    torch.stack(token_ids_masked_list),
                    torch.stack(attention_masked_list),
                    effective_token_ids,
                    len(mask_indices),
                    1,
                )
            )

        return effective_token_ids, masked_tensors

    # adjust for when the word is multiple tokens then -> have the mean of the tokens similar to sentence_scoring
    def per_token_score(self, input_ids, scores, word, reduction):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        scores = scores[0].numpy()

        token_scores = {
            token_id.item(): score for token_id, score in zip(input_ids, scores)
        }

        word_token_ids = self.tokenizer(word)["input_ids"]

        effective_word_token_ids = [
            token_id
            for token_id in word_token_ids
            if token_id != self.sep_token_id
            and token_id != self.cls_token_id
            and token_id != self.pad_token_id
        ]

        word_tensor = torch.tensor(effective_word_token_ids)

        word_token_scores = []
        for word_token_id in word_tensor:
            key = word_token_id.item()
            if key in token_scores:
                word_token_scores.append(token_scores[key])

            else:
                continue

        word_score = list(map(reduction, word_token_scores))

        word_token_scores_tensor = torch.tensor(word_token_scores, dtype=torch.float32)

        word_score_mean = word_token_scores_tensor.sum(0).item()

        return word, word_score_mean

    #contains code from the Kauf et al. repo
    def evaluate_sentence(
        self,
        input,
        method,
        per_token=False,
        return_tensors=False,
        base_two=False,
        prob=False,
    ):
        sentences, word = input

        original_ids, tokenized = self.prepare_texts(sentence=sentences, method=method)

        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*tokenized)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )

        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits.detach()[torch.arange(sum(lengths)), indices]

        logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)

        if base_two:
            logprob_distribution = logprob_distribution / torch.tensor(2).log()

        if prob:
            logprob_distribution = logprob_distribution.exp()

        scores = (
            logprob_distribution[torch.arange(sum(lengths)), effective_token_ids]
            .type(torch.DoubleTensor)
            .split(lengths)
        )
        scores = [s for s in scores]

        reduction = lambda x: x.sum(0).item()
        reduced = list(map(reduction, scores))

        if per_token:
            return self.per_token_score(
                input_ids=original_ids, scores=scores, word=word, reduction=reduction
            )

        else:
            return reduced, scores

    def evaluate_non_gender_mlm(self, max_size=10000):

        df = read_de_neutral()

        texts = df["text"].tolist()
        model = self.model.to(self.device)
        is_generative = self.tokenizer.mask_token is None
        is_llama = "llama" in self.model.name_or_path.lower()
        if is_generative:
            result = evaluate_clm_perplexity(
                model, self.tokenizer, texts[:1000], verbose=False
            )
            # todo gpt
            # if is_llama:
            # else:
            #    result, stats = evaluate_clm(model, tokenizer, texts, verbose=False)
        else:
            result, stats = evaluate_mlm(model, self.tokenizer, texts, verbose=False)

        return result

    def get_seq_score(self, sentence, tokenizer, model, device):
        token_ids_input = [tokenizer.encode(sentence, add_special_tokens=True)]
        tokenized = [tokenizer.convert_ids_to_tokens(tok) for tok in token_ids_input]

        tensor_input = torch.tensor(token_ids_input).to(self.device)  # .unsqueeze(0)
        predictions = model(tensor_input)[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
        loss = loss.item()

        return loss, tokenized

    # Calcuates the whole sentence score (also useful for adjectives which are not tokenized as one word (no need to scrap))
    def get_sentence_score(self, sentences, tokenizer, model, device):
        candidate_probabilities = {}
        cands_tokenized = []

        for sentence in sentences:
            score, tokenized = self.get_seq_score(sentence, tokenizer, model, device)
            candidate_probabilities[sentence] = score
            cands_tokenized.append(tokenized)

        return candidate_probabilities  

    def preprocess(self, sentence, mask, det=True):
        if det:
            return sentence.replace("[ARTICLE]", mask)
        else:
            return sentence.replace("[ADJECTIVE]", mask)

    def evaluate_single_locus_grammaticality(self, df, top_k=None, batch_size=32, method="kauf_l2r"): 
        vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        mask = self.tokenizer.mask_token

        df["masked_det"] = df["masked_det"].apply(
            lambda x: self.preprocess(sentence=x, mask=mask)
        )
        df["masked_ajd"] = df["masked_adj"].apply(
            lambda x: self.preprocess(sentence=x, mask=mask, det=False)
        )

        masked_texts_det = list(df["masked_det"])
        masked_texts_adj = list(df["masked_adj"])

        output_data = {key: [] for key in ["acc_d"]}

        output_data = {
            key: []
            for key in [
                "gram_det_p",
                "ungram_det_p",
                "gram_sent_prob",
                "ungram_sent_prob",
                "gram_det_score",
                "ungram_det_score",
                "gram_adj_score",
                "ugram_adj_score",
                "most_likely_token",
                "candidate_score_certainty",
                "cf_candidate_score_certainty",
                "candidate_det_certainty",
                "cf_candidate_det_certainty",
                "candidate_adj_certainty",
                "cf_candidate_adj_certainty",
                "gender",
            ]
        }
        for start_idx in range(0, len(masked_texts_det), batch_size):
            end_idx = min(start_idx + batch_size, len(masked_texts_det))

            batch_df = df.iloc[start_idx:end_idx]

            batch_texts_det = masked_texts_det[start_idx:end_idx]
            batch_texts_adj = masked_texts_adj[start_idx:end_idx]

            batch_tokenized_text = self.tokenizer(
                batch_texts_det, padding=True, return_tensors="pt", truncation=True
            )
            input_ids = batch_tokenized_text["input_ids"].to(self.device)
            attention_mask = batch_tokenized_text["attention_mask"].to(self.device)

            mask_index = (input_ids == self.tokenizer.mask_token_id)[1]

            with torch.no_grad():
                output_det = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = output_det.logits

            for i in range(len(batch_texts_det)):
                text = batch_texts_det[i]
                label_det = batch_df.loc[batch_df.index[i], "det_label"]
                counter_label_det = batch_df.loc[batch_df.index[i], "cf_det_label"]

                label_adj = batch_df.loc[batch_df.index[i], "adj_label"]
                counter_label_adj = batch_df.loc[batch_df.index[i], "cf_adj_label"]

                gram_gender = batch_df.loc[batch_df.index[i], "gender"]

                f_sentence = batch_df.loc[batch_df.index[i], "candidate"]
                cf_sentence = batch_df.loc[batch_df.index[i], "cf_candidate"]

                candidate_det = (f_sentence, label_det)
                candidate_adj = (f_sentence, label_adj)

                cf_candidate_det = (cf_sentence, counter_label_det)
                cf_candidate_adj = (cf_sentence, counter_label_adj)
                

                candidate_score_prob = self.evaluate_sentence(
                    input=candidate_det,
                    method=method,
                    per_token=False,
                    base_two=False,
                    prob=True,
                )[0][0]
                cf_candidate_score_prob = self.evaluate_sentence(
                    input=cf_candidate_det,
                    method=method,
                    per_token=False,
                    base_two=False,
                    prob=True,
                )[0][0]

                candidate_score_certainty = self.evaluate_sentence(
                    input=candidate_det,
                    method=method,
                    per_token=False,
                    base_two=False,
                    prob=False,
                )[0][0]
                cf_candidate_score_certainty = self.evaluate_sentence(
                    input=cf_candidate_det,
                    method=method,
                    per_token=False,
                    base_two=False,
                    prob=False,
                )[0][0]

                # token scores for factual counterfactual DETERMINERS
                candidate_det_score = self.evaluate_sentence(
                    input=candidate_det,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=True,
                )[1]
                cf_candidate_det_score = self.evaluate_sentence(
                    input=cf_candidate_det,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=True,
                )[1]

                candidate_det_certainty = self.evaluate_sentence(
                    input=candidate_det,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=False,
                )[1]
                cf_candidate_det_certainty = self.evaluate_sentence(
                    input=cf_candidate_det,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=False,
                )[1]

                # token scores for factual counterfactual ADJECTIVES
                candidate_adj_score = self.evaluate_sentence(
                    input=candidate_adj,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=True,
                )[1]
                cf_candidate_adj_score = self.evaluate_sentence(
                    input=cf_candidate_adj,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=True,
                )[1]

                candidate_adj_certainty = self.evaluate_sentence(
                    input=candidate_adj,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=False,
                )[1]
                cf_candidate_adj_certainty = self.evaluate_sentence(
                    input=cf_candidate_adj,
                    method=method,
                    per_token=True,
                    base_two=False,
                    prob=False,
                )[1]

                # for adjectives only use the pll token score...
                predictions_det = logits[i, mask_index]

                probs = softmax(predictions_det, dim=-1).squeeze()

                # TODO get this from the config...
                article_probs = {
                    art: probs[self.tokenizer.convert_tokens_to_ids(art)].item()
                    for art in [
                        "Der",
                        "Des",
                        "Den",
                        "Dem",
                        "Die",
                        "Ein",
                        "Eines",
                        "Einen",
                        "Einem",
                        "Mein",
                        "Meines",
                        "Meinen",
                        "Meinem",
                        "Eine",
                        "Einer",
                        "Meine",
                        "Meiner",
                    ]
                }

                sorted_probs = sorted(
                    article_probs.items(), key=lambda x: x[1], reverse=True
                )
                most_likely_token = sorted_probs[0][0]

                if top_k:
                    top_k = 100
                    top_k_values, top_k_indices = torch.topk(probs, top_k)
                    top_k_tokens = [vocab[int(i)] for i in top_k_indices]
                    top_k_probs = top_k_values.tolist()

                label_prob_det = article_probs[label_det]
                cf_label_prob_det = article_probs[counter_label_det]

                output_data["gram_det_p"].append(label_prob_det)
                output_data["ungram_det_p"].append(cf_label_prob_det)
                output_data["gram_sent_prob"].append(candidate_score_prob)
                output_data["ungram_sent_prob"].append(cf_candidate_score_prob)
                output_data["gram_det_score"].append(candidate_det_score)
                output_data["ungram_det_score"].append(cf_candidate_det_score)
                output_data["gram_adj_score"].append(candidate_adj_score)
                output_data["ugram_adj_score"].append(cf_candidate_adj_score)
                output_data["most_likely_token"].append(most_likely_token)
                output_data["candidate_score_certainty"].append(
                    candidate_score_certainty
                )
                output_data["cf_candidate_score_certainty"].append(
                    cf_candidate_score_certainty
                )
                output_data["candidate_det_certainty"].append(candidate_det_certainty)
                output_data["cf_candidate_det_certainty"].append(
                    cf_candidate_det_certainty
                )
                output_data["candidate_adj_certainty"].append(candidate_adj_certainty)
                output_data["cf_candidate_adj_certainty"].append(
                    cf_candidate_adj_certainty
                )
                output_data["gender"].append(gram_gender)

        output_data_df = pd.DataFrame.from_dict(output_data)

        return compute_metrics(output_data_df)


def evaluate_model_with_ae(
    model_path: str, eval_csv: str, output_csv: str, gender_factors=None, lrs=None
):
    """
    Run agreement and likelihood evaluation for multiple gender factors and learning rates.

    Args:
        model_path (str): Path to pretrained model directory.
        eval_csv (str): Path to CSV with evaluation data.
        output_csv (str): Path where results should be saved.
        gender_factors (list, optional): Gender factors to test. Default spans [-10, -2, -1, ..., 10].
        lrs (list, optional): Learning rates to test. Default spans [-1, -0.95, ..., 1].
    """

    log.info(f"Evaluating {model_path}")
    if gender_factors is None:
        gender_factors = [
            -10,
            -2,
            -1,
            -0.8,
            -0.6,
            -0.4,
            -0.2,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1,
            2,
            10,
        ]
    if lrs is None:
        lrs = [
            -1,
            -0.95,
            -0.9,
            -0.85,
            -0.8,
            -0.75,
            -0.7,
            -0.65,
            -0.6,
            -0.55,
            -0.5,
            0,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1,
        ]

    pairs = {(g_f, lr) for g_f in gender_factors for lr in lrs}

    # Load model + tokenizer
    base_model = ModelWithGradiend.from_pretrained(model_path)
    eval_data = pd.read_csv(eval_csv)

    results = []

    for g_f, lr in pairs:
        # Modify model
        enhanced_model = base_model.modify_model(
            lr=lr, gender_factor=g_f, top_k=None, part="decoder", top_k_part="decoder"
        )

        decoder_analysis = DeDecoderAnalysis(enhanced_model, base_model.tokenizer)

        # Evaluate
        lls = decoder_analysis.evaluate_non_gender_mlm()
        output = decoder_analysis.evaluate_single_locus_grammaticality(eval_data)

        # Add config info
        output["g_f"] = g_f
        output["lr"] = lr
        output["lls"] = lls

        print(output)
        results.append(output)

    results_df = pd.DataFrame(results)
    cols = ["g_f", "lr"] + [c for c in results_df.columns if c not in ["g_f", "lr"]]
    results_df = results_df[cols]

    # Save
    results_df.to_csv(output_csv, index=False)

    return results_df


def diff(a, b):
    return a - b


def compute_metrics(output_data):
    # accuracy as does the model assign a higher prob to the grammatical sent.
    acc_sent_overall = (
        output_data["gram_sent_prob"] > output_data["ungram_sent_prob"]
    ).mean()
    acc_det_score_overall = (
        output_data["gram_det_score"] > output_data["ungram_det_score"]
    ).mean()
    acc_adj_score_overall = (
        output_data["gram_adj_score"] > output_data["ugram_adj_score"]
    ).mean()

    grouped_sent_acc = output_data.groupby("gender").apply(
        lambda x: (x["gram_sent_prob"] > x["ungram_sent_prob"]).mean()
    )
    grouped_det_acc = output_data.groupby("gender").apply(
        lambda x: (x["gram_det_score"] > x["ungram_det_score"]).mean()
    )
    grouped_adj_acc = output_data.groupby("gender").apply(
        lambda x: (x["gram_adj_score"] > x["ugram_adj_score"]).mean()
    )

    output_data["certainty_sent_overall"] = output_data.apply(
        lambda x: diff(
            x["candidate_score_certainty"], x["cf_candidate_score_certainty"]
        ),
        axis=1,
    )
    output_data["certainty_det_overall"] = output_data.apply(
        lambda x: diff(x["candidate_det_certainty"], x["cf_candidate_det_certainty"]),
        axis=1,
    )
    output_data["certainty_adj_overall"] = output_data.apply(
        lambda x: diff(x["candidate_adj_certainty"], x["cf_candidate_adj_certainty"]),
        axis=1,
    )

    grouped_sent_certainty = output_data.groupby("gender")[
        "certainty_sent_overall"
    ].mean()
    grouped_det_certainty = output_data.groupby("gender")[
        "certainty_det_overall"
    ].mean()
    grouped_adj_certainty = output_data.groupby("gender")[
        "certainty_adj_overall"
    ].mean()

    prob_grouped = output_data.groupby("gender")["gram_det_p"].mean()

    return {
        "acc_sent_overall": acc_sent_overall,
        "acc_det_score_overall": acc_det_score_overall,
        "acc_adj_score_overall": acc_adj_score_overall,
        "acc_f_sent": grouped_sent_acc.get("Fem", None),
        "acc_m_sent": grouped_sent_acc.get("Masc", None),
        "acc_f_det": grouped_det_acc.get("Fem", None),
        "acc_m_det": grouped_det_acc.get("Masc", None),
        "acc_f_adj": grouped_adj_acc.get("Fem", None),
        "acc_m_adj": grouped_adj_acc.get("Masc", None),
        "certainty_sent_overall": output_data["certainty_sent_overall"].mean(),
        "certainty_det_overall": output_data["certainty_det_overall"].mean(),
        "certainty_adj_overall": output_data["certainty_adj_overall"].mean(),
        "certainty_sent_f": grouped_sent_certainty.get("Fem", None),
        "certianty_sent_m": grouped_sent_certainty.get("Masc", None),
        "certainty_det_f": grouped_det_certainty.get("Fem", None),
        "certainty_det_m": grouped_det_certainty.get("Masc", None),
        "certainty_adj_f": grouped_adj_certainty.get("Fem", None),
        "certainty_adj_m": grouped_adj_certainty.get("Masc", None),
        "prob_m": prob_grouped.get("Masc"),
        "prob_f": prob_grouped.get("Fem"),
    }


from minicons import scorer
if __name__ == "__main__":
    default_evaluation_gender_factors = [-1]
    default_evaluation_lrs = [-5e-2]

    model_scorer = scorer.MaskedLMScorer('bert-base-german-cased', 'gpu' if torch.cuda.is_available() else 'cpu')

    # distilbert_model_path = "results/experiments/gradiend/MF/3e-05/distilbert-base-german-cased/1"
    # pairs = {(gender_factor, lr) for gender_factor in default_evaluation_gender_factors for lr in default_evaluation_lrs}
    # bert_with_ae = ModelWithGradiend.from_pretrained(distilbert_model_path)
    # eval_data = pd.read_csv("masked_NP.csv")

    distilbert_model_path = (
        "results/experiments/gradiend/MF/1e-05/bert-base-german-cased/0"
    )
    pairs = {
        (gender_factor, lr)
        for gender_factor in default_evaluation_gender_factors
        for lr in default_evaluation_lrs
    }
    bert_with_ae = ModelWithGradiend.from_pretrained(distilbert_model_path)
    eval_data = pd.read_csv("gradiend/data/der_die_das/eval/masked_NP.csv")

    evaluate_model_with_ae(
        model_path=distilbert_model_path,
        eval_csv="gradiend/data/der_die_das/eval/masked_NP.csv",
        output_csv="masked_NP_results.csv",
        gender_factors=default_evaluation_gender_factors,
        lrs=default_evaluation_lrs,
    )

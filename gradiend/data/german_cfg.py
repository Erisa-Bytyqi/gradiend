import os
from nltk import grammar, parse
from nltk.parse.generate import generate
import pandas as pd
import spacy
import pandas as pd
from tqdm import tqdm
import csv
from pathlib import Path
from DERBI.derbi import DERBI


nlp = spacy.load("de_core_news_sm")


derbi = DERBI(nlp)

NounPhrase_MF = grammar.CFG.fromstring(
    """
% start S 
S -> NP_DEF '.' | NP_INDEF '.'
NP_DEF -> ART_DEF_m ADJ_DEF_m NN_m | ART_DEF_f ADJ_DEF_f NN_f 
NP_INDEF -> ART_INDEF_m ADJ_INDEF_m NN_m | ART_INDEF_f ADJ_DEF_f NN_f 
ART_DEF_m -> 'Der' | 'Dieser'
ADJ_DEF_m -> 'dunkle' | 'helle' | 'runde' 
ART_INDEF_m -> 'Ein' | 'Mein'
ADJ_INDEF_m -> 'dunkler' | 'heller' | 'runder' 
ART_DEF_f -> 'Die' | 'Diese'
ADJ_DEF_f -> 'dunkle' | 'helle' | 'schwarze'
ART_INDEF_f -> 'Eine' | 'Meine'
NN_m -> 'Raum' | 'Ton' | 'Fleck' | 'Klang'
NN_f -> 'Wolke'| 'Farbe' | 'Hose' | 'Jacke' 
"""
)

counters = {
    "Case=Nom|Definite=Def|Gender=Masc|Number=Sing|PronType=Art": "Die",
    "Case=Nom|Definite=Def|Gender=Fem|Number=Sing|PronType=Art": "Der",
    "Case=Nom|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art": "Eine",
    "Case=Nom|Definite=Ind|Gender=Fem|Number=Sing|PronType=Art": "Ein",
    "Case=Nom|Gender=Masc|Number=Sing|PronType=Dem": "Diese",
    "Case=Nom|Gender=Fem|Number=Sing|PronType=Dem": "Dieser",
    "Case=Nom|Gender=Fem|Number=Sing|Poss=Yes|PronType=Prs": "Mein",
    "Case=Nom|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs": "Meine",
}


def create_simple_sents():
    """
    NOT IN USE. Create simple sentences using the defined CFG.
    """
    result = []
    for sentence in generate(NounPhrase_MF):
        candidate = {}
        masked_sentence = sentence[:]
        counter_sentence = sentence[:]
        masked_sentence[0] = "[ARTICLE]"
        og_sentence = (" ").join(sentence)

        candidate["text"] = og_sentence
        candidate["masked"] = (" ").join(masked_sentence)

        doc = nlp(og_sentence)

        if doc[0] and counters.get(str(doc[0].morph)):
            label = doc[0].morph.get("Gender")
            pron_type = doc[0].morph.get("PronType")
            counter_sentence[0] = counters[str(doc[0].morph)]

        else:
            continue

        candidate = {
            "text": og_sentence,
            "masked": (" ").join(masked_sentence),
            "cf": (" ").join(counter_sentence),
            "label": label,
            "pron_type": pron_type,
        }

        result.append(candidate)

    cfg_result = pd.DataFrame(result)

    cfg_result.to_csv("NP_cfg.csv")


def process_composite(
    input_file: str,
    output_file: str,
    counters: dict,
    failed_pairs_file: str = None,
):
    """Process composite phrases. Only the nominative is used."""

    composite = pd.read_csv(input_file)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if failed_pairs_file:
        os.makedirs(os.path.dirname(failed_pairs_file), exist_ok=True)

    results = []
    failed_pairs = []

    m_names, f_names, n_names = [], [], []

    ARTICLE_MAP = {
        "Masc": {"def": "Der", "indef": "Ein", "poss": "Mein"},
        "Fem": {"def": "Die", "indef": "Eine", "poss": "Meine"},
        "Neut": {"def": "Das", "indef": "Ein", "poss": "Mein"},
    }

    def build_phrases(phrase: str, gender: str):
        articles = ARTICLE_MAP[gender]
        return [
            f"{articles['def']} {phrase}",
            f"{articles['indef']} {phrase}",
            f"{articles['poss']} {phrase}",
        ]

    def process_sentence(og_sentence: str):
        """Run derbi + counterfactual sentence creation, return candidate dict or nothing."""
        candidate = {}

        masked_sentence = og_sentence.split(" ")
        counter_sentence = og_sentence.split(" ")
        masked_sentence[0] = "[ARTICLE]"

        candidate["text"] = og_sentence
        candidate["masked"] = (" ").join(masked_sentence)

        doc = nlp(og_sentence)

        if doc[0] and counters.get(str(doc[0].morph)):
            gender = doc[0].morph.get("Gender")
            pron_type = doc[0].morph.get("PronType")
            if doc[0].morph.get("Definite"):
                definite = doc[0].morph.get("Definite")[0]
            else:
                definite = None
            cf_label = counters[str(doc[0].morph)]
            counter_sentence[0] = cf_label

            return {
                "text": og_sentence,
                "masked": (" ").join(masked_sentence),
                "cf": (" ").join(counter_sentence),
                "label": doc[0],
                "cf_label": cf_label,
                "gender": gender[0],
                "pron_type": pron_type[0],
                "definite": definite,
                "case": "Nom",
            }

        else:
            return None  # Skip if no valid morph found

    with open(output_file, "w", newline="", encoding="utf-8") as outf:

        result_writer = csv.DictWriter(
            outf,
            fieldnames=[
                "text",
                "masked",
                "cf",
                "label",
                "cf_label",
                "gender",
                "pron_type",
                "definite",
                "case",
            ],
        )
        result_writer.writeheader()

        if failed_pairs_file:
            with open(failed_pairs_file, "w", newline="", encoding="utf-8") as failf:
                fail_writer = csv.writer(failf)
                fail_writer.writerow(["adjective", "noun"])

        for phrase, adj, noun in tqdm(
            zip(composite["phrase"], composite["adjective"], composite["noun"]),
            total=len(composite["phrase"]),
            desc="Processing phrases",
        ):
            try:
                noun_gender = nlp(noun)[0].morph.get("Gender")[0]

                phrases = build_phrases(phrase, noun_gender)

                # declinate w. derbi
                derbi_sentences = [
                    derbi(
                        phrases[0],
                        [
                            {
                                "Case": "Nom",
                                "Declination": "Weak",
                                "Gender": noun_gender,
                                "Number": "Sing",
                            }
                        ],
                        [1],
                    ),
                    derbi(
                        phrases[1],
                        [
                            {
                                "Case": "Nom",
                                "Declination": "Strong",
                                "Gender": noun_gender,
                                "Number": "Sing",
                            }
                        ],
                        [1],
                    ),
                    derbi(
                        phrases[2],
                        [
                            {
                                "Case": "Nom",
                                "Declination": "Strong",
                                "Gender": noun_gender,
                                "Number": "Sing",
                            }
                        ],
                        [1],
                    ),
                ]

                for sent in derbi_sentences:
                    candidate = process_sentence(sent)
                    if candidate:

                        result_writer.writerow(candidate)

            # DERBI exceptions that occur when an adjective from the dataset is tagged as a different part of speech.
            except (ValueError, IndexError, AttributeError):
                if failed_pairs_file:
                    fail_writer.writerow((adj, noun))
                continue



def process_declensions(
    input_file: str,
    output_file: str,
    counter_type: str,
    failed_pairs_file: str = None,
):
    """
    Process declensions for all cases (Nom, Gen, Acc, Dat) based on the output of process_composite().

    counter_type: one of {"MF", "MN", "FN"}
    """

    base_data = pd.read_csv(input_file)
    # Valid case set
    cases = ["Nom", "Gen", "Acc", "Dat"]

    # Counter gender swap rules
    COUNTER_MAP = {
        "MF": {"Masc": "Fem", "Fem": "Masc"},
        "MN": {"Masc": "Neut", "Neut": "Masc"},
        "FN": {"Fem": "Neut", "Neut": "Fem"},
    }

    if counter_type not in COUNTER_MAP:
        raise ValueError(
            f"Invalid counter_type '{counter_type}'. Must be one of {list(COUNTER_MAP.keys())}."
        )

    # Prepare output CSV (append mode). Write header only if file does not exist or is empty.
    results_fieldnames = [
        "text",
        "candidate",
        "cf_candidate",
        "case",
        "gender",
        "cf_gender",
        "pron_type",
        "def",
        "masked_det",
        "masked_adj",
        "det_label",
        "cf_det_label",
        "adj_label",
        "cf_adj_label",
    ]
    results_file_exists = (
        os.path.exists(output_file) and os.path.getsize(output_file) > 0
    )
    results_fh = open(output_file, "a", encoding="utf-8", newline="")
    results_writer = csv.DictWriter(results_fh, fieldnames=results_fieldnames)
    if not results_file_exists:
        results_writer.writeheader()

    if failed_pairs_file:
        failed_exists = (
            os.path.exists(failed_pairs_file) and os.path.getsize(failed_pairs_file) > 0
        )
        failed_fh = open(failed_pairs_file, "a", encoding="utf-8", newline="")
        failed_writer = csv.writer(failed_fh)
        if not failed_exists:
            failed_writer.writerow(["text", "case"])
    else:
        failed_fh = None
        failed_writer = None

    def declinate_art_adj(sample):
        text = sample["text"]
        gender = sample["gender"]
        pron_type = sample["pron_type"]
        art_def = sample["definite"]

        for case in cases:
            try:
                declination = "Weak" if case in ["Nom", "Acc"] else "Mixed"

                det_declension = {"Gender": gender, "Case": case}
                adj_declension = {
                    "Case": case,
                    "Declination": declination,
                    "Gender": gender,
                    "Number": "Sing",
                }

                cf_gender = COUNTER_MAP[counter_type].get(gender)
                if not cf_gender:
                    # If gender not mappable for some reason, skip this case.
                    continue

                cf_det_declension = {"Gender": cf_gender, "Case": case}
                cf_adj_declension = {
                    "Case": case,
                    "Declination": "Mixed",
                    "Gender": cf_gender,
                    "Number": "Sing",
                }

                if case == "Gen":
                    noun_declension = {"Case": "Gen"}
                    candidate = derbi(
                        text,
                        [det_declension, adj_declension, noun_declension],
                        [0, 1, 2],
                    )
                    cf_candidate = derbi(
                        text,
                        [cf_det_declension, cf_adj_declension, noun_declension],
                        [0, 1, 2],
                    )
                else:
                    candidate = derbi(text, [det_declension, adj_declension], [0, 1])
                    cf_candidate = derbi(
                        text, [cf_det_declension, cf_adj_declension], [0, 1]
                    )

                masked_cands = _create_masked_candidates(candidate, cf_candidate)
                # Write the row immediately
                results_writer.writerow(
                    {
                        "text": text,
                        "candidate": candidate,
                        "cf_candidate": cf_candidate,
                        "case": case,
                        "gender": gender,
                        "cf_gender": cf_gender,
                        "pron_type": pron_type,
                        "def": art_def,
                        **masked_cands,
                    }
                )
            except Exception:
             
                if failed_writer:
                    failed_writer.writerow([text, case])

    # Process each row and stream results
    for _, sample in tqdm(
        base_data.iterrows(), total=len(base_data), desc="Processing"
    ):
        declinate_art_adj(sample)

    # Close open file handles
    results_fh.flush()
    results_fh.close()
    if failed_fh:
        failed_fh.flush()
        failed_fh.close()

    # Return a DataFrame loaded from the results file (keeps memory usage low during processing)
    return pd.read_csv(output_file)


def _create_masked_candidates(candidate: str, cf_candidate: str):
    """
    From 'candidate' and 'cf_candidate' strings like 'Der dunkle Raum' and 'Die dunkle Raum',
    compute masked variants [ARTICLE] dunkle Raum and Der [ADJECTIVE] Raum.
    """

    split_sent = candidate.split()
    split_cf = cf_candidate.split()


    det_label = split_sent[0] if len(split_sent) > 0 else ""
    adj_label = split_sent[1] if len(split_sent) > 1 else ""
    cf_det_label = split_cf[0] if len(split_cf) > 0 else ""
    cf_adj_label = split_cf[1] if len(split_cf) > 1 else ""


    masked_det_tokens = split_sent.copy()
    masked_adj_tokens = split_sent.copy()
    if len(masked_det_tokens) > 0:
        masked_det_tokens[0] = "[ARTICLE]"
    if len(masked_adj_tokens) > 1:
        masked_adj_tokens[1] = "[ADJECTIVE]"

    masked_det = " ".join(masked_det_tokens) if split_sent else ""
    masked_adj = " ".join(masked_adj_tokens) if split_sent else ""

    return {
        "masked_det": masked_det,
        "masked_adj": masked_adj,
        "det_label": det_label,
        "cf_det_label": cf_det_label,
        "adj_label": adj_label,
        "cf_adj_label": cf_adj_label,
    }


if __name__ == "__main__":

    counters_MF = {
        "Case=Nom|Definite=Def|Gender=Masc|Number=Sing|PronType=Art": "Die",
        "Case=Nom|Definite=Def|Gender=Fem|Number=Sing|PronType=Art": "Der",
        "Case=Nom|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art": "Eine",
        "Case=Nom|Definite=Ind|Gender=Fem|Number=Sing|PronType=Art": "Ein",
        "Case=Nom|Gender=Masc|Number=Sing|PronType=Dem": "Diese",
        "Case=Nom|Gender=Fem|Number=Sing|PronType=Dem": "Dieser",
        "Case=Nom|Gender=Fem|Number=Sing|Poss=Yes|PronType=Prs": "Mein",
        "Case=Nom|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs": "Meine",
    }

    counters_MN = {
        "Case=Nom|Definite=Def|Gender=Masc|Number=Sing|PronType=Art": "Das",
        "Case=Nom|Definite=Def|Gender=Neut|Number=Sing|PronType=Art": "Der",
        "Case=Nom|Gender=Masc|Number=Sing|PronType=Dem": "Dieses",
        "Case=Nom|Gender=Neut|Number=Sing|PronType=Dem": "Dieser",
    }

    counters_FN = {
        "Case=Nom|Definite=Def|Gender=Neut|Number=Sing|PronType=Art": "Die",
        "Case=Nom|Definite=Def|Gender=Fem|Number=Sing|PronType=Art": "Das",
        "Case=Nom|Definite=Ind|Gender=Neut|Number=Sing|PronType=Art": "Eine",
        "Case=Nom|Definite=Ind|Gender=Fem|Number=Sing|PronType=Art": "Ein",
        "Case=Nom|Gender=Neut|Number=Sing|PronType=Dem": "Diese",
        "Case=Nom|Gender=Fem|Number=Sing|PronType=Dem": "Dieses",
        "Case=Nom|Gender=Fem|Number=Sing|Poss=Yes|PronType=Prs": "Mein",
        "Case=Nom|Gender=Neut|Number=Sing|Poss=Yes|PronType=Prs": "Meine",
    }

    input_file_composite = "gradiend/data/der_die_das/eval/decoder/deu_adj_n.csv"
    output_nom_noun_phrases = "gradiend/data/der_die_das/eval/decoder/FN/test.csv"

    input_file_declension = output_nom_noun_phrases
    output_file_declension = (
        "gradiend/data/der_die_das/eval/decoder/FN/deu_adj_n_declension.csv"
    )

    # process_composite(
    #     input_file=input_file,
    #     output_file=output_nom_noun_phrases,
    #     counters=counters_FN,
    #     failed_pairs_file=None,
    # )

    process_declensions(
        input_file=input_file_declension,
        output_file=output_file_declension,
        counter_type="FN",
        failed_pairs_file=None,
    )

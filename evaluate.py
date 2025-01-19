import string

import pandas as pd
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm


def process_txt(text):
  """Remove punctuation and strip the text."""
  translator = str.maketrans('', '', string.punctuation)
  return text.translate(translator).strip()

def load_benchmark(token: str) -> pd.DataFrame:
    """IMPORTANT: The dataset is gated. 
    You need to access the dataset first from the Hugging Face Hub."""
    benchmark = load_dataset('atlasia/TerjamaBench', token=token)
    benchmark = benchmark['test']
    return pd.DataFrame(benchmark)

def evaluate(references: list[str], predictions: list[str]) -> list[dict]:
    """Evaluate the predictions using BLEU, CHRF, and TER metrics.
    Returns a list of dictionaries containing the scores.
    If the evaluation fails, the score will be set to None."""
    assert len(references) == len(predictions), "The number of references and predictions should be the same."
    
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    ter = TER()

    results = []
    for i in tqdm(range(len(references)), desc="Evaluating samples"):
        refs = [process_txt(references[i])]
        hyp = process_txt(predictions[i]) 
        try:
            bleu_score = bleu.sentence_score(references=refs, hypothesis=hyp).score
            chrf_score = chrf.sentence_score(references=refs, hypothesis=hyp).score
            ter_score =  ter.sentence_score(references=refs, hypothesis=hyp).score
            results.append({'BLEU': bleu_score, 'CHRF': chrf_score, 'TER': ter_score})
        except Exception as e:
            print(e)
            results.append({'BLEU': None, 'CHRF': None, 'TER': None})
            continue

    return results

def evaluate_model(references: list[str], predictions: list[str]) -> dict:
    """Avg the scores of the references and predictions."""
    scores = evaluate(references, predictions)
    scores_df = pd.DataFrame(scores)
    return scores_df.mean().to_dict()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pickle as pkl
# import evaluate
from rouge_score import rouge_scorer
import math
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from .metric import *
from .plot import *
from .umwp_eval import *
import _settings

USE_Roberta = False
USE_EXACT_MATCH = False

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
if USE_Roberta:
    SenSimModel = SentenceTransformer(os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'))

def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()

def getAcc(resultDict, file_name):
    correctCount = 0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if "coqa" in file_name or "TruthfulQA" in file_name:
            additional_answers = item["additional_answers"]
            rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores))
        if rougeScore>0.5:
            correctCount += 1
    print("Acc:", 1.0*correctCount/len(resultDict))

def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]

def load_umwp_answerable_mapping(umwp_dataset_file):
    """
    Load the UMWP dataset and create a mapping from ID to answerable status.
    
    Args:
        umwp_dataset_file: Path to the UMWP dataset file
        
    Returns:
        Dictionary mapping from ID to answerable status
    """
    print(f"Loading UMWP dataset from {umwp_dataset_file}...")
    with open(umwp_dataset_file, 'r', encoding='utf-8') as f:
        umwp_data =[]
        for line in f:
            item = json.loads(line.strip())
            umwp_data.append(item)
    
    id_to_answerable = {}
    for item in umwp_data:
        # print(item)
        item_id = item.get('id', None)
        if item_id is not None:
            id_to_answerable[item_id] = item.get('answerable', False)
            # print(item_id, item.get('answerable', False))
            # break
    
    print(f"Loaded {len(id_to_answerable)} items with answerable field from UMWP dataset")
    return id_to_answerable

def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    # HSIC = []
    # HSIC_mid = []
    # HSIC_second_last =[]
    num_hsic_scores = len(resultDict[0]["hsic_scores"])
    # Initialize a list to hold the HSIC scores for each item at each layer/hyperparameter
    all_hsic_scores = [[] for _ in range(num_hsic_scores)]
    # if 'umwp' in file_name:
    #     umwp_dataset_file = "/home/yash/eigenscore/data/datasets/StandardDataset.jsonl"  
    #     umwp_answerable_mapping = load_umwp_answerable_mapping(umwp_dataset_file)

    for item in resultDict:
        # print(item)
        # if 'umwp' in file_name:
        #     if item['answerable'][0]:
        #         continue
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # HSIC_mid.append(0 if math.isnan(item["hsic_mid"]) else item["hsic_mid"])
        # HSIC_second_last.append(0 if math.isnan(item["hsic_second_last"]) else item["hsic_second_last"])
        if item['hsic_scores'] == 0:
            item['hsic_scores'] = [0]*33
        for i, hsic_score in enumerate(item["hsic_scores"]):
            # all_hsic_scores[i].append(0 if math.isnan(hsic_score) else hsic_score)
            all_hsic_scores[i].append(hsic_score)

        if 'umwp' in file_name:
            item_id = item['id']
            answerable = item['answerable'][0]
            # print(answerable)
            # answer = None
            extracted_number = None
            pred_unanswerable = judge_generated_text_unanswerable(generations)
            # print(id, pred_unanswerable)
            generations = generations.replace('\n', ' ')
            # print('hello1')
            # Create a copy of the item and add prediction
            # output_item = item.copy()
            # output_item['prediction'] = pred_unanswerable
            

            # if answerable == True:
            #     answerable_cnt += 1
            # else:
            #     unanswerable_cnt += 1

            if pred_unanswerable == False:
                last_sentence = extract_last_few_sentences(generations)
                extracted_number = [float(item[0]) for item in extract_number(last_sentence)]
                # print(id, last_sentence, extracted_number, answer, answerable_correct)
            else:
                pass

            if answerable == False:
                if pred_unanswerable == True:
                    # TP += 1
                    # print('hello2')
                    Label.append(1)
                    # print('TP', id)
                else:
                    # print('hello3')
                    Label.append(0)
                    # FN += 1

            elif answerable == True and pred_unanswerable == True:
                # FP += 1
                # print('hello4')
                Label.append(0)

            if answerable == True:
                if pred_unanswerable == False:
                    # answer = item['answer'][0]
                    if ansGT in extracted_number:
                        # output_item['gen_correct_answer']=True
                        # Acc += 1
                        # print('hello5')
                        Label.append(1)
                    else:
                        # print('hello6')
                        # output_item['gen_correct_answer']=False
                        Label.append(0)
        elif USE_Roberta:
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ansGT, SenSimModel) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity>0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        elif USE_EXACT_MATCH:
            similarity = compute_exact_match(generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [compute_exact_match(generations, ansGT) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity==1:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        else:
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            if rougeScore>0.5:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)

    thresholds_list = []
    
    # Calculate AUROC for each HSIC score list
    for i, hsic_score_list in enumerate(all_hsic_scores):
        fpr, tpr, thresholds = roc_curve(Label, hsic_score_list)
        AUROC = auc(fpr, tpr)
        threshold = get_threshold(thresholds, tpr, fpr)
        thresholds_list.append(threshold)
        print(f"AUROC-HSIC score {i}:", AUROC)

    if 'umwp' not in file_name:
        for i, hsic_score_list in enumerate(all_hsic_scores):
            rho_hsic = getPCC(Score, hsic_score_list)
            print(f"rho_HSIC_{i}:", rho_hsic)


    if "TruthfulQA" in file_name:
        for i, hsic_score_list in enumerate(all_hsic_scores):
            acc = getTruthfulQAAccuracy(Label, hsic_score_list, thresholds_list[i])
            print(f"TruthfulQA HSIC_{i} Accuracy:", acc)


def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt

def getTruthfulQAAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


if __name__ == "__main__":
    file_name = "/data/output/ablations/llama3-8b_SQuAD_1/0.pkl"
    print(file_name)
    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    if 'umwp' not in file_name:
        getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)


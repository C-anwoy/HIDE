import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    Perplexity = []
    Energy = []
    HSIC = []
    HSIC_mid = []
    HSIC_second_last =[]
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []


    for item in resultDict:
        # print(item)
        if 'umwp' in file_name:
            if item['answerable'][0]:
                continue
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        # Perplexity.append(-item["perplexity"])
        # Energy.append(-item["energy"])
        HSIC.append(item["hsic"])
        # HSIC_mid.append(item["hsic_mid_keywords"])
        # HSIC_second_last.append(item["hsic_second_last_keyword"])
        # HSIC_mid.append(0 if math.isnan(item["hsic_mid"]) else item["hsic_mid"])
        # HSIC_second_last.append(0 if math.isnan(item["hsic_second_last"]) else item["hsic_second_last"])
        # Entropy.append(-item["entropy"])
        # LexicalSimilarity.append(item["lexical_similarity"])
        # SentBertScore.append(-item["sent_bertscore"])
        # EigenIndicator.append(-item["eigenIndicator"])
        # EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])

        if 'umwp' in file_name:
            item_id = item['id']
            # print(item_id.type)
            # print(item_id)
            # print(umwp_answerable_mapping)
            # print(f"test: {umwp_answerable_mapping.get(1)}")
            # answerable = umwp_answerable_mapping.get(int(item_id), False)

            # if answerable != item['answerable'][0]:
            #     print(item_id)
            #     print(item['prompt'])
            #     print(answerable)
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


    fpr, tpr, thresholds = roc_curve(Label, HSIC)
    AUROC = auc(fpr, tpr)
    thresh_HSIC = get_threshold(thresholds, tpr, fpr)
    print("AUROC-HSIC:", AUROC)


    if 'umwp' not in file_name:
        # rho_Perplexity = getPCC(Score, Perplexity)
        # rho_Entropy = getPCC(Score, Entropy)
        # rho_Energy = getPCC(Score, Energy)
        rho_HSIC = getPCC(Score, HSIC)
        # rho_HSIC_mid = getPCC(Score, HSIC_mid)
        # rho_HSIC_second_last = getPCC(Score, HSIC_second_last)
        # rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
        # rho_EigenIndicator = getPCC(Score, EigenIndicator)
        # rho_EigenIndicatorOutput = getPCC(Score, EigenIndicatorOutput)
        # print("rho_Perplexity:", rho_Perplexity)
        # print("rho_Energy:", rho_Energy)
        print("rho_HSIC:", rho_HSIC)
        # print("rho_HSIC:", rho_HSIC_mid)
        # print("rho_HSIC:", rho_HSIC_second_last)
        # print("rho_Entropy:", rho_Entropy)
        # print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
        # print("rho_EigenScore:", rho_EigenIndicator)
        # print("rho_EigenScoreOutput:", rho_EigenIndicatorOutput)

    if "TruthfulQA" in file_name:
        # acc = getTruthfulQAAccuracy(Label, Perplexity, thresh_Perplexity)
        # print("TruthfulQA Perplexity Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, Energy, thresh_Energy)
        # print("TruthfulQA Energy Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, HSIC, thresh_HSIC)
        print("TruthfulQA HSIC Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, HSIC_mid, thresh_HSIC_mid)
        # print("TruthfulQA HSIC_mid Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, HSIC_second_last, thresh_HSIC_second_last)
        # print("TruthfulQA HSIC_second_last Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, Entropy, thresh_Entropy)
        # print("TruthfulQA Entropy Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
        # print("TruthfulQA LexicalSimilarity Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, SentBertScore, thresh_SentBertScore)
        # print("TruthfulQA SentBertScore Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, EigenIndicator, thresh_EigenScore)
        # print("TruthfulQA EigenIndicator Accuracy:", acc)
        # acc = getTruthfulQAAccuracy(Label, EigenIndicatorOutput, thresh_EigenScoreOutput)
        # print("TruthfulQA EigenIndicatorOutput Accuracy:", acc)

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
    file_name = "/data/output/llama3-8b-instruct_triviaqa_6/0.pkl"
    print(file_name)
    f = open(file_name, "rb")
    resultDict = pkl.load(f)
    # printInfo(resultDict)
    if 'umwp' not in file_name:
        getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)


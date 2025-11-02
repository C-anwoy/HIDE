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
import _settings

USE_Roberta = True
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


def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    Perplexity = []
    Energy = []
    HSIC = []
    # HSIC_mid = []
    # HSIC_second_last =[]
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        Perplexity.append(-item["perplexity"])
        Energy.append(-item["energy"])
        HSIC.append(item["hsic"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        SentBertScore.append(-item["sent_bertscore"])
        EigenIndicator.append(-item["eigenIndicator"])
        EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])

        if USE_Roberta:
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

    print(len(Label), len(Perplexity))
    fpr, tpr, thresholds = roc_curve(Label, Perplexity)
    AUROC = auc(fpr, tpr)
    # thresh_Perplexity = thresholds[np.argmax(tpr - fpr)]
    thresh_Perplexity = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Perplexity:", AUROC)
    # print("thresh_Perplexity:", thresh_Perplexity)
    # VisAUROC(tpr, fpr, AUROC, "Perplexity")

    fpr, tpr, thresholds = roc_curve(Label, Energy)
    AUROC = auc(fpr, tpr)
    # thresh_Energy = thresholds[np.argmax(tpr - fpr)]
    thresh_Energy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Energy:", AUROC)
    # print("thresh_Energy:", thresh_Energy)
    # VisAUROC(tpr, fpr, AUROC, "Energy")

    fpr, tpr, thresholds = roc_curve(Label, HSIC)
    AUROC = auc(fpr, tpr)
    # thresh_HSIC = thresholds[np.argmax(tpr - fpr)]
    thresh_HSIC = get_threshold(thresholds, tpr, fpr)
    print("AUROC-HSIC:", AUROC)
    # print("thresh_HSIC:", thresh_HSIC)
    # VisAUROC(tpr, fpr, AUROC, "HSIC")

    fpr, tpr, thresholds = roc_curve(Label, Entropy)
    AUROC = auc(fpr, tpr)
    # thresh_Entropy = thresholds[np.argmax(tpr - fpr)]
    thresh_Entropy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Entropy:", AUROC)
    # print("thresh_Entropy:", thresh_Entropy)
    # VisAUROC(tpr, fpr, AUROC, "NormalizedEntropy")

    fpr, tpr, thresholds = roc_curve(Label, LexicalSimilarity)
    AUROC = auc(fpr, tpr)
    # thresh_LexicalSim = thresholds[np.argmax(tpr - fpr)]
    thresh_LexicalSim = get_threshold(thresholds, tpr, fpr)
    print("AUROC-LexicalSim:", AUROC)
    # print("thresh_LexicalSim:", thresh_LexicalSim)
    # VisAUROC(tpr, fpr, AUROC, "LexicalSim")

    fpr, tpr, thresholds = roc_curve(Label, SentBertScore)
    AUROC = auc(fpr, tpr)
    # thresh_SentBertScore = thresholds[np.argmax(tpr - fpr)]
    thresh_SentBertScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-SentBertScore:", AUROC)
    # print("thresh_SentBertScore:", thresh_SentBertScore)
    # VisAUROC(tpr, fpr, AUROC, "SentBertScore")

    fpr, tpr, thresholds = roc_curve(Label, EigenIndicator)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScore = thresholds[np.argmax(tpr - fpr)]
    thresh_EigenScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-EigenScore:", AUROC)
    # print("thresh_EigenScore:", thresh_EigenScore)
    # VisAUROC(tpr, fpr, AUROC, "EigenScore", file_name.split("_")[1])

    fpr, tpr, thresholds = roc_curve(Label, EigenIndicatorOutput)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScoreOutput = thresholds[np.argmax(tpr - fpr)]
    thresh_EigenScoreOutput = get_threshold(thresholds, tpr, fpr)
    print("AUROC-EigenScore-Output:", AUROC)
    # print("thresh_EigenScoreOutput:", thresh_EigenScoreOutput)
    # VisAUROC(tpr, fpr, AUROC, "EigenScoreOutput", file_name.split("_")[1])

    if 'umwp' not in file_name:
        rho_Perplexity = getPCC(Score, Perplexity)
        rho_Entropy = getPCC(Score, Entropy)
        rho_Energy = getPCC(Score, Energy)
        rho_HSIC = getPCC(Score, HSIC)
        rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
        rho_EigenIndicator = getPCC(Score, EigenIndicator)
        rho_EigenIndicatorOutput = getPCC(Score, EigenIndicatorOutput)
        print("rho_Perplexity:", rho_Perplexity)
        print("rho_Energy:", rho_Energy)
        print("rho_Entropy:", rho_Entropy)
        print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
        print("rho_EigenScore:", rho_EigenIndicator)
        print("rho_HSIC:", rho_HSIC)
        print("rho_EigenScoreOutput:", rho_EigenIndicatorOutput)


    if "TruthfulQA" in file_name:
        acc = getTruthfulQAAccuracy(Label, Perplexity, thresh_Perplexity)
        print("TruthfulQA Perplexity Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, Energy, thresh_Energy)
        print("TruthfulQA Energy Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, HSIC, thresh_HSIC)
        print("TruthfulQA HSIC Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, Entropy, thresh_Entropy)
        print("TruthfulQA Entropy Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
        print("TruthfulQA LexicalSimilarity Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, SentBertScore, thresh_SentBertScore)
        print("TruthfulQA SentBertScore Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, EigenIndicator, thresh_EigenScore)
        print("TruthfulQA EigenIndicator Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, EigenIndicatorOutput, thresh_EigenScoreOutput)
        print("TruthfulQA EigenIndicatorOutput Accuracy:", acc)

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

def get_inference_time(resultDict):
    greedy_generation_time=[]
    multiple_generation_time=[]
    perplexity_time=[]
    energy_time=[]
    hsic_time=[]
    entropy_time=[]
    lexical_sim_time=[]
    bert_score_time=[]
    eigen_time=[]
    eigen_output_time=[]

    for item in resultDict:
        greedy_generation_time.append(item["greedy_generation_time"])
        multiple_generation_time.append(item["multiple_generation_time"])
        perplexity_time.append(item["perplexity_time"])
        energy_time.append(item["energy_time"])
        hsic_time.append(item["hsic_time"])
        entropy_time.append(item["entropy_time"])
        lexical_sim_time.append(item["lexical_sim_time"])
        bert_score_time.append(item["bert_score_time"])
        eigen_time.append(item["eigen_time"])
        eigen_output_time.append(item["eigen_output_time"])
    
    # Calculate average times
    avg_greedy_generation_time = sum(greedy_generation_time) / len(greedy_generation_time)
    avg_multiple_generation_time = sum(multiple_generation_time) / len(multiple_generation_time)
    avg_perplexity_time = sum(perplexity_time) / len(perplexity_time)
    avg_energy_time = sum(energy_time) / len(energy_time)
    avg_hsic_time = sum(hsic_time) / len(hsic_time)
    avg_entropy_time = sum(entropy_time) / len(entropy_time)
    avg_lexical_sim_time = sum(lexical_sim_time) / len(lexical_sim_time)
    avg_bert_score_time = sum(bert_score_time) / len(bert_score_time)
    avg_eigen_time = sum(eigen_time) / len(eigen_time)
    avg_eigen_output_time = sum(eigen_output_time) / len(eigen_output_time)

    # Print results
    print("Average Inference Times (seconds per question):")
    print(f"Greedy Generation: {avg_greedy_generation_time}")
    print(f"Multiple Generation: {avg_multiple_generation_time}")
    print(f"Perplexity: {(avg_perplexity_time+avg_greedy_generation_time)}")
    print(f"Energy: {(avg_energy_time+avg_greedy_generation_time)}")
    print(f"Entropy: {(avg_entropy_time+avg_multiple_generation_time)}")
    print(f"Lexical Similarity: {(avg_lexical_sim_time+avg_multiple_generation_time)}")
    print(f"BERT Score: {(avg_bert_score_time+avg_multiple_generation_time)}")
    print(f"Eigen Score: {(avg_eigen_time+avg_multiple_generation_time)}")
    print(f"HSIC: {(avg_hsic_time+avg_greedy_generation_time)}")
    print(f"Eigen Output: {(avg_eigen_output_time+avg_multiple_generation_time)}")

if __name__ == "__main__":
    # file_name = "/home/anwoy/HIDE/data/output/llama3-8b_SQuAD_1/0.pkl"
    # file_name = "/home/anwoy/HIDE/data/output/llama3-8b_nq_open_1/0.pkl"
    # file_name = "/home/anwoy/HIDE/data/output/gemma2_SQuAD_1/0.pkl"
    file_name = "/home/anwoy/HIDE/data/output/gemma2_nq_open_1/0.pkl"

    print(file_name)
    f = open(file_name, "rb")
    resultDict = pkl.load(f)
    # printInfo(resultDict)
    getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)
    get_inference_time(resultDict)


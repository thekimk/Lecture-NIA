# System related and data input controls
import os

# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Understanding of Data
from ydata_profiling import ProfileReport
import missingno as msno

# Common
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler
import random

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from xgboost import plot_importance as plot_importance_xgb
from lightgbm import plot_importance as plot_importance_lgbm
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
import shap

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Embedding, Reshape, RepeatVector, Permute, Multiply, Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers import Conv3D, Convolution3D, MaxPooling3D, AveragePooling3D, UpSampling3D
# from keras.layers import TimeDistributed, CuDDN, CuDNNLSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras_tqdm import tqdm_callback, TQDMNotebookCallback
import keras.backend as K

# Text
import re
import string
from ast import literal_eval
import kss    # 문장분리
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, LdaModel, LdaMulticore, CoherenceModel
from keybert import KeyBERT
## 영어
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as sw_eng
## 한국어
from konlpy.tag import Hannanum, Kkma, Komoran, Okt, Mecab
from kss import split_sentences
from kiwipiepy import Kiwi
from spacy.lang.ko.stop_words import STOP_WORDS as sw_kor
from soynlp.normalizer import *
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor
from soynlp.tokenizer import LTokenizer
from sentence_transformers import SentenceTransformer

from datasets import Features, Value, ClassLabel
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from sklearn.model_selection import train_test_split, TimeSeriesSplit


### Date and Author: 20250328, Kyungwon Kim ###
### simple descriptive statistics by class label
def table_ratiobyclass(df, Y_colname, label_list='auto', 
                       precision='{:.2f}', significance=0.05,
                       sorting=False, save=False):
    # setting
    from scipy.stats import ttest_ind, chi2_contingency
    if label_list == 'auto':
        label_list = [str(i) for i in np.unique(df[Y_colname])]
    X_colname = [x for x in df.columns if x != Y_colname]

    # cross table
    index_multi, count_bylabel, stat_list, stat_sort, sigval_counting = [], pd.DataFrame(), [], [], 0
    for col in X_colname:
        sub_table = pd.crosstab(index=df[col], columns=df[Y_colname])

        ## drop rows containing zero value & chi2 calculation
        sub_table = sub_table.loc[(sub_table!=0).all(axis=1)]

        if sub_table.shape[0] == 0:
            continue
        elif sub_table.shape[0] == 1:
            stat, pval = ttest_ind(df.loc[list(df.index[df[Y_colname] == 0]), col], 
                                   df.loc[list(df.index[df[Y_colname] == 1]), col])
            stat_reformat = ['$T$='+str(precision.format(stat))+', $p$='+str(precision.format(pval))]
        else:
            stat, pval, dof, _ = chi2_contingency([sub_table.values[:,0], sub_table.values[:,1]])
            stat_reformat = [r'$\chi^2$='+str(precision.format(stat))+', $p$='+str(precision.format(pval))] + ['']*(sub_table.shape[0]-1)
        if pval <= significance: 
            sigval_counting = sigval_counting + 1 # 유의수준 5% 이상 갯수 카운팅
        stat_list.extend(stat_reformat)
        stat_sort.extend([stat]*sub_table.shape[0])
        ## rearrange
        index_multi.extend([(col, idx) for idx in sub_table.index])
        count_bylabel = pd.concat([count_bylabel, sub_table], axis=0)
    count_bylabel.columns = label_list
    print('Number of significant features: ', str(sigval_counting) + ' (' + str(sigval_counting / len(X_colname)) + '%)')

    # ratio table
    ratio_bylabel = count_bylabel.div(count_bylabel.sum(axis=1), axis=0) * 100
    ratio_bylabel.columns = label_list

    # rearrange
    df_bylabel = pd.DataFrame()
    for i in range(len(label_list)):
        df_bylabel[label_list[i]] = count_bylabel[label_list[i]].astype(str) + ' (' + ratio_bylabel[label_list[i]].map(precision.format).astype(str) + '%)'
    df_bylabel.index = pd.MultiIndex.from_tuples(index_multi)
    df_bylabel['Statistics'] = stat_list
    if sorting:
        df_bylabel['Col_sorting'] = stat_sort
        df_bylabel.reset_index(inplace=True)
        df_bylabel.sort_values(by=['Col_sorting', 'level_1'], ascending=[False, True], inplace=True)   
        index_multi = [(df_bylabel['level_0'][i], df_bylabel['level_1'][i]) for i in df_bylabel.index]
        df_bylabel.index = pd.MultiIndex.from_tuples(index_multi)
        df_bylabel = df_bylabel.iloc[:,2:-1]
    ## 저장
    if save:
        from datetime import datetime
        time_now = datetime.now().strftime('%Y%m%d')
        save_name = os.path.join(os.getcwd(),'Result','DescriptiveStatistics_Binary_'+time_now+'.csv')
        df_bylabel.reset_index().to_csv(save_name, index=False, encoding='utf-8-sig')

    return df_bylabel


### Date and Author: 20250328, Kyungwon Kim ###
### MDIS data preprocessing
def preprocessing_MDIS_KK(df):
    df_prep = df.copy()

    # 전처리 1: 결측치의 비율이 50%를 초과하는 변수 삭제
    missing_ratio = df_prep.isnull().mean()
    cols_to_drop_missing = missing_ratio[missing_ratio > 0.5].index
    df_prep.drop(cols_to_drop_missing, axis=1, inplace=True)

    # 전처리 2: 값의 종류가 1개인 변수 삭제
    cols_to_drop_single = [col for col in df_prep.columns if df_prep[col].nunique() == 1]
    df_prep.drop(cols_to_drop_single, axis=1, inplace=True)

    # 전처리 3: 분석과 무관한 변수 삭제
    irrelevant_cols = ['가구일련번호', '가구원번호', '가구주관계코드', '가구가중값', '가구원가중값']
    df_prep.drop([col for col in irrelevant_cols if col in df_prep.columns], axis=1, inplace=True)

    # 전처리 4: 결측치 처리
    for col in df_prep.columns:
        if df_prep[col].isnull().sum() > 0:
            if df_prep[col].dtype == 'object':
                df_prep[col].fillna('Temp_KK', inplace=True)
            else:
                df_prep[col].fillna(df_prep[col].max() + 1, inplace=True)

    # 전처리 5: 종속변수 설정
    Y = df_prep[['기부여부']]
    X = df_prep.drop('기부여부', axis=1)

    # 전처리 6: 데이터 분할
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    # 전처리 7: MinMaxScaler 적용
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train.values, X_test.values, Y_train, Y_test, df_prep


### Date and Author: 20250328, Kyungwon Kim ###
def evaluation_class_ConfusionMatrix(Y_true, Y_pred, label_list=None):
    # calculation
    conf_mat = metrics.confusion_matrix(Y_true, Y_pred)
    
    # visualization
    names = ['TN', 'FP', 'FN', 'TP']
    counts = ['{0:0.0f}'.format(value) for value in conf_mat.flatten()]
    percentages = ['{0:.4%}'.format(value) for value in conf_mat.flatten()/np.sum(conf_mat)]
    labels = np.asarray([f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]).reshape(2,2)
    if label_list == None:
        index_name = ['True '+str(i) for i in np.unique(np.array(Y_true))]
        column_name = ['Pred '+str(i) for i in np.unique(np.array(Y_true))]
        sns.set(font_scale=1.4)
        sns.heatmap(conf_mat, annot=labels, fmt='', 
                    cmap='Blues', linewidths=.5, annot_kws={"size": 14}, 
                    xticklabels=column_name, yticklabels=index_name)
        plt.show()
        conf_mat = pd.DataFrame(conf_mat, index=index_name, columns=column_name)
    else:
        sns.set(font_scale=1.5)
        sns.heatmap(conf_mat, annot=labels, fmt='', 
                    cmap='Blues', linewidths=1, annot_kws={"size": 14},
                    xticklabels=label_list, yticklabels=label_list)
        plt.show()
        conf_mat = pd.DataFrame(conf_mat, index=label_list, columns=label_list)
    
    return conf_mat


### Date and Author: 20250328, Kyungwon Kim ###
def evaluation_class_Metrics(Y_true, P_pred, label='Test'):
    Y_pred = (P_pred >= 0.5).astype(int)
    
    # metrics
    conf_mat = metrics.confusion_matrix(Y_true, Y_pred)
    N = Y_true.shape[0]
    TP, TN = str('{:.0f}'.format(conf_mat[1,1])), str('{:.0f}'.format(conf_mat[0,0]))
    FP, FN = str('{:.0f}'.format(conf_mat[0,1])), str('{:.0f}'.format(conf_mat[1,0]))
    precision = metrics.precision_score(Y_true, Y_pred)
    recall = metrics.recall_score(Y_true, Y_pred, pos_label=1)
    specificity = metrics.recall_score(Y_true, Y_pred, pos_label=0)
    f1_score = metrics.f1_score(Y_true, Y_pred)
    accuracy = metrics.accuracy_score(Y_true, Y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(Y_true, Y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(Y_true, P_pred)
    auc = metrics.auc(fpr, tpr)
    
    # rearrange
    metric_summary = pd.DataFrame([N, TP, TN, FP, FN, 
                                   precision, recall, specificity, f1_score, 
                                   accuracy, balanced_accuracy, auc],
                                  index=['N', 'True Positive', 'True Negative', 'False Positive', 'False Negative',
                                         'Precision', 'Recall', 'Specificity', 'F1-score', 'Accuracy', 'Balanced Accuracy', 'AUC'],
                                  columns=[label])
    
    return metric_summary.T


### Date and Author: 20250328, Kyungwon Kim ###
def evaluation_class_ROCAUC(Y_true, P_pred, figsize=(10,5), label='Logistic Regression'):
    fpr, tpr, thresholds = roc_curve(Y_true, P_pred)
    cm = metrics.confusion_matrix(Y_true, P_pred>=0.5)
    recall = cm[1,1] / cm.sum(axis=1)[1]
    fallout = cm[0,1] / cm.sum(axis=1)[0]
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot([fallout], [recall], 'ro', ms=10)
    plt.title('AUC: ' + str(auc(fpr, tpr)), fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid('on')
    plt.show()
    

### Date and Author: 20250328, Kyungwon Kim ###
def prediction_class(model, X_train, Y_train, X_test, Y_test,
                     LABEL_LIST=None, ALGO_NAME='Classification'):
    # 예측하기
    ## Y vector
    if Y_train.shape[1] == 1:
        try:
            P_trpred = pd.DataFrame(model.predict_proba(X_train)[:,-1], 
                                    index=Y_train.index, columns=['Pred'])
            P_tepred = pd.DataFrame(model.predict_proba(X_test)[:,-1], 
                                    index=Y_test.index, columns=['Pred'])
        except:
            P_trpred = pd.DataFrame(model.predict(X_train), 
                                    index=Y_train.index, columns=['Pred'])
            P_tepred = pd.DataFrame(model.predict(X_test), 
                                    index=Y_test.index, columns=['Pred'])
    ## Y matrix
    else:
        P_trpred = pd.DataFrame(model.predict(X_train)[:,-1], columns=['Pred'])
        P_tepred = pd.DataFrame(model.predict(X_test)[:,-1], columns=['Pred'])
        ### Y 정답 정리
        Y_train = pd.DataFrame(np.argmax(Y_train, axis=-1), columns=['Real'])
        Y_test = pd.DataFrame(np.argmax(Y_test, axis=-1), columns=['Real'])        
    ## 라벨 부여
    Y_trpred = (P_trpred >= 0.5).astype(int)
    Y_tepred = (P_tepred >= 0.5).astype(int)

    # 평가/검증
    print('Performance: ')
    ## Confusion Matrix
    evaluation_class_ConfusionMatrix(Y_test, Y_tepred, label_list=LABEL_LIST)
    ## ROC Curve + AUC
    evaluation_class_ROCAUC(Y_test, P_tepred, label=ALGO_NAME)
    ## Classification Metrics
    Score_te = evaluation_class_Metrics(Y_test, P_tepred, label='Test set')
    Score_trte = evaluation_class_Metrics(pd.concat([Y_train, Y_test], axis=0), 
                                          np.concatenate((P_trpred, P_tepred)), 
                                          label='Entire population')
    ## 저장
    Scores = pd.concat([Score_te, Score_trte], axis=0).reset_index()
    Scores.index = [ALGO_NAME, ALGO_NAME]
    Scores = Scores.reset_index().rename(columns = {'index':'Dataset', 'level_0':'Algorithm'})
    save_name = os.path.join(os.getcwd(),'Result','Performance_'+ALGO_NAME+'.csv')
    Scores.to_csv(save_name, index=False, encoding='utf-8-sig')    
    
    return Score_te, Score_trte


def explanation_SHAP_KK(model, X_train, X_test, X_colname,
                        MAX_DISPLAY=10, model_type='linear',
                        link='logit', sample_size=1,
                        sample_size_1000=1000,
                        plot_interaction=True,
                        feature_display_range=None):
    """
    SHAP 설명을 제공하는 함수.

    Parameters:
    - model: 학습된 모델 (예: XGBClassifier, LogisticRegression)
    - X_train: 훈련 데이터 (특징 행렬)
    - X_test: 테스트 데이터
    - X_colname: 특성 이름 리스트 (예: ['feature1', 'feature2', ...])
    - MAX_DISPLAY: SHAP 설명에서 표시할 최대 특성 수 (기본값 10)
    - model_type: 모델 유형 ('linear', 'tree', 'svm', 'auto' 등, 기본값 'linear')
    - link: 로짓 링크 함수 (기본값 'logit')
    - sample_size: 개별 설명을 위한 샘플 크기 (기본값 1)
    - sample_size_1000: 1000개의 샘플에 대해 설명할 때 샘플 크기 (기본값 1000)
    - plot_interaction: 상호작용 설명을 플로팅할지 여부 (기본값 True)
    - feature_display_range: 특성 표시 범위 (기본값 None)
    """

    # SHAP 설명자 생성
    explainer = shap.Explainer(model, X_train, algorithm=model_type, feature_names=X_colname)
    shap_values_train = explainer(X_train)
    shap_values_test = explainer(X_test)

    # 개별 설명 (샘플 1) for Train
    shap_sample_train = shap_values_train.sample(sample_size)
    print("Train Sample Explanation:")
    shap.decision_plot(base_value=shap_sample_train.base_values,
                       shap_values=shap_sample_train.values,
                       features=shap_sample_train.data,
                       feature_names=X_colname,
                       feature_display_range=feature_display_range or slice(None, -MAX_DISPLAY, -1),
                       link=link, highlight=0)
    shap.initjs()
    display(shap.force_plot(base_value=shap_sample_train.base_values,
                            shap_values=shap_sample_train.values,
                            features=shap_sample_train.data,
                            feature_names=X_colname,
                            link=link))

    # 개별 설명 (샘플 1) for Test
    shap_sample_test = shap_values_test.sample(sample_size)
    print("Test Sample Explanation:")
    shap.decision_plot(base_value=shap_sample_test.base_values,
                       shap_values=shap_sample_test.values,
                       features=shap_sample_test.data,
                       feature_names=X_colname,
                       feature_display_range=feature_display_range or slice(None, -MAX_DISPLAY, -1),
                       link=link, highlight=0)
    shap.initjs()
    display(shap.force_plot(base_value=shap_sample_test.base_values,
                            shap_values=shap_sample_test.values,
                            features=shap_sample_test.data,
                            feature_names=X_colname,
                            link=link))

    # 개별 설명 (샘플 1000) for Train
    shap_sample_train = shap_values_train.sample(sample_size_1000)
    print("Train Sample 1000 Explanation:")
    shap.initjs()
    display(shap.force_plot(base_value=shap_sample_train.base_values,
                            shap_values=shap_sample_train.values,
                            features=shap_sample_train.data,
                            feature_names=X_colname,
                            link=link))

    # 개별 설명 (샘플 1000) for Test
    shap_sample_test = shap_values_test.sample(sample_size_1000)
    print("Test Sample 1000 Explanation:")
    shap.initjs()
    display(shap.force_plot(base_value=shap_sample_test.base_values,
                            shap_values=shap_sample_test.values,
                            features=shap_sample_test.data,
                            feature_names=X_colname,
                            link=link))

    # 전체 설명 (beeswarm 플롯) for Train
    print("Train Total Explanation (Beeswarm):")
    shap.plots.beeswarm(shap_values=shap_values_train, max_display=MAX_DISPLAY)

    # 전체 설명 (beeswarm 플롯) for Test
    print("Test Total Explanation (Beeswarm):")
    shap.plots.beeswarm(shap_values=shap_values_test, max_display=MAX_DISPLAY)

    # 상호작용 설명 (dependence plot) for Train
    if plot_interaction:
        print("Train Total Explanation by Interaction:")
        shap_importance_train = np.abs(shap_values_train.values).mean(axis=0)
        feature_order_train = np.argsort(shap_importance_train)[::-1]
        feature_order_train = [X_colname[i] for i in feature_order_train]

        for col in feature_order_train[:MAX_DISPLAY]:
            shap.dependence_plot(ind=col, shap_values=shap_values_train.values,
                                 features=X_train, feature_names=X_colname)

        # 상호작용 설명 (dependence plot) for Test
        print("Test Total Explanation by Interaction:")
        shap_importance_test = np.abs(shap_values_test.values).mean(axis=0)
        feature_order_test = np.argsort(shap_importance_test)[::-1]
        feature_order_test = [X_colname[i] for i in feature_order_test]

        for col in feature_order_test[:MAX_DISPLAY]:
            shap.dependence_plot(ind=col, shap_values=shap_values_test.values,
                                 features=X_test, feature_names=X_colname)


### Date and Author: 20250104, Kyungwon Kim ###
### Data reshaping for deep learning Y to OneHotEncoding-Y
def reshape_YtoOneHot(Y_train, Y_test):
    # 변환
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(Y_train)
    Y_tr_reshape = ohe.transform(Y_train)
    Y_te_reshape = ohe.transform(Y_test)

    # 정리
    columns_ohe = [Y_train.columns[0]+'_'+str(i) for i in np.unique(Y_train)]
    Y_tr_reshape = pd.DataFrame(Y_tr_reshape, columns=columns_ohe)
    Y_te_reshape = pd.DataFrame(Y_te_reshape, columns=columns_ohe)
    print('Y_train_reshape:', Y_train.shape, '-->', Y_tr_reshape.shape)
    print('Y_test_reshape:', Y_test.shape, '-->', Y_te_reshape.shape)

    return Y_tr_reshape, Y_te_reshape


### Date and Author: 20250104, Kyungwon Kim ###
### Data reshaping for deep learning X to 3D-X
def reshape_X2Dto3D(X_train, X_test):
    # 변환
    X_tr_reshape = X_train.reshape(-1, X_train.shape[1], 1)
    X_te_reshape = X_test.reshape(-1, X_test.shape[1], 1)
    print('X_train_reshape:', X_train.shape, '-->', X_tr_reshape.shape)
    print('X_test_reshape:', X_test.shape, '-->', X_te_reshape.shape)

    return X_tr_reshape, X_te_reshape


### Date and Author: 20250409, Kyungwon Kim ###
### df highlighting
def table_highlight(df_target, colidx=[], axis=0, minmax='max'):
    # 하위 함수 정의
    def highlight_max_red_bold(x):
        return ['font-weight: bold; background-color: red' if i == x.max() else '' for i in x]
    def highlight_min_red_bold(x):
        return ['font-weight: bold; background-color: red' if i == x.min() else '' for i in x]

    # 대상 colname 선택
    if colidx == []:
        colname = list(df_target.columns)
    else:
        colname = [list(df_target.columns)[i] for i in colidx if i < len(list(df_target.columns))]

    # 스타일 적용
    if minmax == 'max':
        df_styled = df_target.style.apply(highlight_max_red_bold, subset=colname, axis=axis)
    elif minmax == 'min':
        df_styled = df_target.style.apply(highlight_min_red_bold, subset=colname, axis=axis)

    return df_styled


### Date and Author: 20230802, Kyungwon Kim ###
### Concat Prediction Scores for Target Algorithms
def prediction_summary(folder_location, algonames=None, 
                       colidx=[], axis=0, highlight_direct='max', save_name='Performance.csv'):
    if algonames == None:
        print('Please Select the Algorithm Names... for Cancatenating Results!')
    else:
        # 성능측정 파일들
        files = [i for i in os.listdir(folder_location) if i.startswith('Performance_')]
        
        # 원하는 알고리즘에 맞게 데이터 로딩 및 결합
        scores_te, scores_trte = pd.DataFrame(), pd.DataFrame()
        for algo in algonames:
            for file in files:
                if algo == file.split('_')[1].split('.')[0]:
                    score = pd.read_csv(os.path.join(folder_location, file))
                    scores_te = pd.concat([scores_te, score.iloc[[0],:]], axis=0)
                    scores_trte = pd.concat([scores_trte, score.iloc[[1],:]], axis=0)
        
    # 정리
    scores_te = scores_te[[scores_te.columns[1], scores_te.columns[0]]+list(scores_te.columns[2:])].reset_index().iloc[:,1:]
    scores_trte = scores_trte[[scores_trte.columns[1], scores_trte.columns[0]]+list(scores_trte.columns[2:])].reset_index().iloc[:,1:]
    display(table_highlight(scores_te, colidx=colidx, axis=axis, minmax=highlight_direct), 
            table_highlight(scores_trte, colidx=colidx, axis=axis, minmax=highlight_direct))
    scores = pd.concat([scores_te, scores_trte], axis=0)
    save_name = os.path.join(os.getcwd(),'Result',save_name)
    scores.to_csv(save_name, index=False, encoding='utf-8-sig')    


### Date and Author: 20240301, Kyungwon Kim ###
### 1개의 문장에 대해서 불필요한 것들 제거하는 기초 전처리
def text_preprocessor(text, language='korean', del_number=False, del_bracket_content=False, stop_words=[]):
    # 한글 맞춤법과 띄어쓰기 체크 (PyKoSpacing, Py-Hanspell)
    # html 태그 제거하기
    text_new = re.sub(r'<[^>]+>', '', str(text))
    # 괄호와 내부문자 제거하기
    if del_bracket_content:
        text_new = re.sub(r'\([^)]*\)', '', text_new)
        text_new = re.sub(r'\[[^)]*\]', '', text_new)
        text_new = re.sub(r'\<[^)]*\>', '', text_new)
        text_new = re.sub(r'\{[^)]*\}', '', text_new)
    else:
        # 괄호 제거하기
        text_new = re.sub(r'\(*\)*', '', text_new)
        text_new = re.sub(r'\[*\]*', '', text_new)
        text_new = re.sub(r'\<*\>*', '', text_new)
        text_new = re.sub(r'\{*\}*', '', text_new)
    # 따옴표 제거하기
    text_new = text_new.replace('"', '')
    text_new = text_new.replace("'", '')
    # 영어(소문자화), 한글, 숫자만 남기고 제거하기
    text_new = re.sub('[^ A-Za-z0-9가-힣]', '', text_new.lower())
    # 한글 자음과 모음 제거하기
    text_new = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', text_new)
    # 숫자 제거하기
    if del_number:
        text_new = re.sub(r'\d+', '', text_new)
    # 숫자를 문자로 인식하기
    text_new = ' '.join([str(word) for word in text_new.split(' ')])
    # 양쪽공백 제거하기
    text_new = text_new.strip()
    # 문장구두점 제거하기
    translator = str.maketrans('', '', string.punctuation)
    text_new = text_new.translate(translator)
    # 2개 이상의 반복글자 줄이기
    text_new = ' '.join([emoticon_normalize(word, num_repeats=2) for word in text_new.split(' ')])
    text_new = ' '.join([repeat_normalize(word, num_repeats=2) for word in text_new.split(' ')])
    # 영어 및 한글 stopwords 제거하기
    stop_words_eng = set(stopwords.words('english'))
    stop_words_kor = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', 
 '로', '에게', '뿐이다', '의거하여', '근거하여', '입각하여', '기준으로', '예하면', '예를 들면', '예를 들자면', '저', '소인', 
 '소생', '저희', '지말고', '하지마', '하지마라', '다른', '물론', '또한', '그리고', '비길수 없다', '해서는 안된다', '뿐만 아니라', 
 '만이 아니다', '만은 아니다', '막론하고', '관계없이', '그치지 않다', '그러나', '그런데', '하지만', '든간에', '논하지 않다',
 '따지지 않다', '설사', '비록', '더라도', '아니면', '만 못하다', '하는 편이 낫다', '불문하고', '향하여', '향해서', '향하다',
 '쪽으로', '틈타', '이용하여', '타다', '오르다', '제외하고', '이 외에', '이 밖에', '하여야', '비로소', '한다면 몰라도', '외에도',
 '이곳', '여기', '부터', '기점으로', '따라서', '할 생각이다', '하려고하다', '이리하여', '그리하여', '그렇게 함으로써', '하지만',
 '일때', '할때', '앞에서', '중에서', '보는데서', '으로써', '로써', '까지', '해야한다', '일것이다', '반드시', '할줄알다',
 '할수있다', '할수있어', '임에 틀림없다', '한다면', '등', '등등', '제', '겨우', '단지', '다만', '할뿐', '딩동', '댕그', '대해서',
 '대하여', '대하면', '훨씬', '얼마나', '얼마만큼', '얼마큼', '남짓', '여', '얼마간', '약간', '다소', '좀', '조금', '다수', '몇',
 '얼마', '지만', '하물며', '또한', '그러나', '그렇지만', '하지만', '이외에도', '대해 말하자면', '뿐이다', '다음에', '반대로',
 '반대로 말하자면', '이와 반대로', '바꾸어서 말하면', '바꾸어서 한다면', '만약', '그렇지않으면', '까악', '툭', '딱', '삐걱거리다',
 '보드득', '비걱거리다', '꽈당', '응당', '해야한다', '에 가서', '각', '각각', '여러분', '각종', '각자', '제각기', '하도록하다',
 '와', '과', '그러므로', '그래서', '고로', '한 까닭에', '하기 때문에', '거니와', '이지만', '대하여', '관하여', '관한', '과연',
 '실로', '아니나다를가', '생각한대로', '진짜로', '한적이있다', '하곤하였다', '하', '하하', '허허', '아하', '거바', '와', '오',
 '왜', '어째서', '무엇때문에', '어찌', '하겠는가', '무슨', '어디', '어느곳', '더군다나', '하물며', '더욱이는', '어느때', '언제',
 '야', '이봐', '어이', '여보시오', '흐흐', '흥', '휴', '헉헉', '헐떡헐떡', '영차', '여차', '어기여차', '끙끙', '아야', '앗',
 '아야', '콸콸', '졸졸', '좍좍', '뚝뚝', '주룩주룩', '솨', '우르르', '그래도', '또', '그리고', '바꾸어말하면', '바꾸어말하자면',
 '혹은', '혹시', '답다', '및', '그에 따르는', '때가 되어', '즉', '지든지', '설령', '가령', '하더라도', '할지라도', '일지라도',
 '지든지', '몇', '거의', '하마터면', '인젠', '이젠', '된바에야', '된이상', '만큼어찌됏든', '그위에', '게다가', '점에서 보아',
 '비추어 보아', '고려하면', '하게될것이다', '일것이다', '비교적', '좀', '보다더', '비하면', '시키다', '하게하다', '할만하다',
 '의해서', '연이서', '이어서', '잇따라', '뒤따라', '뒤이어', '결국', '의지하여', '기대여', '통하여', '자마자', '더욱더',
 '불구하고', '얼마든지', '마음대로', '주저하지 않고', '곧', '즉시', '바로', '당장', '하자마자', '밖에 안된다', '하면된다',
 '그래', '그렇지', '요컨대', '다시 말하자면', '바꿔 말하면', '즉', '구체적으로', '말하자면', '시작하여', '시초에', '이상', '허',
 '헉', '허걱', '바와같이', '해도좋다', '해도된다', '게다가', '더구나', '하물며', '와르르', '팍', '퍽', '펄렁', '동안', '이래',
 '하고있었다', '이었다', '에서', '로부터', '까지', '예하면', '했어요', '해요', '함께', '같이', '더불어', '마저', '마저도',
 '양자', '모두', '습니다', '가까스로', '하려고하다', '즈음하여', '다른', '다른 방면으로', '해봐요', '습니까', '했어요',
 '말할것도 없고', '무릎쓰고', '개의치않고', '하는것만 못하다', '하는것이 낫다', '매', '매번', '들', '모', '어느것', '어느',
 '로써', '갖고말하자면', '어디', '어느쪽', '어느것', '어느해', '어느 년도', '라 해도', '언젠가', '어떤것', '어느것', '저기',
 '저쪽', '저것', '그때', '그럼', '그러면', '요만한걸', '그래', '그때', '저것만큼', '그저', '이르기까지', '할 줄 안다',
 '할 힘이 있다', '너', '너희', '당신', '어찌', '설마', '차라리', '할지언정', '할지라도', '할망정', '할지언정', '구토하다',
 '게우다', '토하다', '메쓰겁다', '옆사람', '퉤', '쳇', '의거하여', '근거하여', '의해', '따라', '힘입어', '그', '다음', '버금',
 '두번째로', '기타', '첫번째로', '나머지는', '그중에서', '견지에서', '형식으로 쓰여', '입장에서', '위해서', '단지', '의해되다',
 '하도록시키다', '뿐만아니라', '반대로', '전후', '전자', '앞의것', '잠시', '잠깐', '하면서', '그렇지만', '다음에', '그러한즉',
 '그런즉', '남들', '아무거나', '어찌하든지', '같다', '비슷하다', '예컨대', '이럴정도로', '어떻게', '만약', '만일',
 '위에서 서술한바와같이', '인 듯하다', '하지 않는다면', '만약에', '무엇', '무슨', '어느', '어떤', '아래윗', '조차', '한데',
 '그럼에도 불구하고', '여전히', '심지어', '까지도', '조차도', '하지 않도록', '않기 위하여', '때', '시각', '무렵', '시간',
 '동안', '어때', '어떠한', '하여금', '네', '예', '우선', '누구', '누가 알겠는가', '아무도', '줄은모른다', '줄은 몰랏다',
 '하는 김에', '겸사겸사', '하는바', '그런 까닭에', '한 이유는', '그러니', '그러니까', '때문에', '그', '너희', '그들', '너희들',
 '타인', '것', '것들', '너', '위하여', '공동으로', '동시에', '하기 위하여', '어찌하여', '무엇때문에', '붕붕', '윙윙', '나',
 '우리', '엉엉', '휘익', '윙윙', '오호', '아하', '어쨋든', '만 못하다하기보다는', '차라리', '하는 편이 낫다', '흐흐', '놀라다',
 '상대적으로 말하자면', '마치', '아니라면', '쉿', '그렇지 않으면', '그렇지 않다면', '안 그러면', '아니었다면', '하든지', '아니면',
 '이라면', '좋아', '알았어', '하는것도', '그만이다', '어쩔수 없다', '하나', '일', '일반적으로', '일단', '한켠으로는', '오자마자',
 '이렇게되면', '이와같다면', '전부', '한마디', '한항목', '근거로', '하기에', '아울러', '하지 않도록', '않기 위해서', '이르기까지',
 '이 되다', '로 인하여', '까닭으로', '이유만으로', '이로 인하여', '그래서', '이 때문에', '그러므로', '그런 까닭에', '알 수 있다',
 '결론을 낼 수 있다', '으로 인하여', '있다', '어떤것', '관계가 있다', '관련이 있다', '연관되다', '어떤것들', '에 대해', '이리하여',
 '그리하여', '여부', '하기보다는', '하느니', '하면 할수록', '운운', '이러이러하다', '하구나', '하도다', '다시말하면', '다음으로',
 '에 있다', '에 달려 있다', '우리', '우리들', '오히려', '하기는한데', '어떻게', '어떻해', '어찌됏어', '어때', '어째서', '본대로',
 '자', '이', '이쪽', '여기', '이것', '이번', '이렇게말하자면', '이런', '이러한', '이와 같은', '요만큼', '요만한 것',
 '얼마 안 되는 것', '이만큼', '이 정도의', '이렇게 많은 것', '이와 같다', '이때', '이렇구나', '것과 같이', '끼익', '삐걱', '따위',
 '와 같은 사람들', '부류의 사람들', '왜냐하면', '중의하나', '오직', '오로지', '에 한하다', '하기만 하면', '도착하다',
 '까지 미치다', '도달하다', '정도에 이르다', '할 지경이다', '결과에 이르다', '관해서는', '여러분', '하고 있다', '한 후', '혼자',
 '자기', '자기집', '자신', '우에 종합한것과같이', '총적으로 보면', '총적으로 말하면', '총적으로', '대로 하다', '으로서', '참',
 '그만이다', '할 따름이다', '쿵', '탕탕', '쾅쾅', '둥둥', '봐', '봐라', '아이야', '아니', '와아', '응', '아이', '참나', '년',
 '월', '일', '령', '영', '일', '이', '삼', '사', '오', '육', '륙', '칠', '팔', '구', '이천육', '이천칠', '이천팔', '이천구',
 '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '령', '영']
    # 분석가 stopwords 추가
    if stop_words != []:
        stop_words_kor.extend(stop_words)
    text_new = ' '.join([word for word in text_new.split(' ') if word not in stop_words_eng])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in stop_words_kor])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in sw_eng])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in sw_kor])
    # 공백이 여러개인 경우 1개로 변경
    text_new = re.sub(r'\s+', ' ', text_new)

    # 한글과 영어 분리저장
    if language == 'korean':
        text_new = re.sub(r'[a-zA-Z]', '', text_new)
    elif language == 'english':
        text_new = re.sub(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', '', text_new)
    
    return text_new


### Date and Author: 20240531, Kyungwon Kim ###
### input change by nixtra package
def convert_df2nixdf(df, Y_colname, X_colname=None, time_colname=None):
    # nixtla 형태반영
    df_nixtra = pd.DataFrame()
    if time_colname == None:
        df_nixtra['ds'] = df.index
        df_nixtra['y'] = df[Y_colname].reset_index().iloc[:,1:].copy()
    else:
        df_nixtra['ds'] = df[time_colname].copy()
        df_nixtra['y'] = df[Y_colname].copy()
                
    # ID 설정
    df_nixtra['unique_id'] = 1.0
    
    # 정리
    df_nixtra = df_nixtra[['unique_id', 'ds', 'y']]
    
    # X 반영
    if X_colname != None:
        df_nixtra = pd.concat([df_nixtra, df[X_colname].reset_index().iloc[:,1:]], axis=1)
    
    return df_nixtra

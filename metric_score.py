import pandas as pd
from scipy.stats import spearmanr, pearsonr
def metric(pscores_file,gscores_file,truth_dir):
    # 读取包含预测分数的CSV文件

    df_predicted = pd.read_csv(pscores_file)
    predicted_scores_dict = dict(zip(df_predicted['filename'], df_predicted['score']))

    # 读取包含标准分数的CSV文件
    df_groudtruth= pd.read_csv(gscores_file)

    # 按照标准分数文件中的视频名称顺序来重新排列预测分数列表
    sorted_pscores = [predicted_scores_dict.get(name) for name in df_groudtruth['filename']]
    sorted_gscores = df_groudtruth['score'].tolist()

    SROCC=spearmanr(sorted_gscores, sorted_pscores)[0]#logit_poor
    PLCC=pearsonr(sorted_gscores, sorted_pscores)[0]
    # print("SROCC:{}, PLCC:{}".format(SROCC,PLCC))

    xlsx = pd.ExcelFile(truth_dir+'/rank-pair-val.xlsx')
    gt_labels_non_source=[]
    pr_labels_non_source=[]
    gt_labels_source=[]
    pr_labels_source=[]
    for sheet_pair in xlsx.sheet_names:
        df_pairs = pd.read_excel(xlsx, sheet_name=sheet_pair)
        for index, row in df_pairs.iterrows():
            video_name1 = row[0]
            video_name2 = row[1]
            video_rank = row[2]
            # 获取对应的预测分数
            video_score1 = predicted_scores_dict.get(video_name1)
            video_score2 = predicted_scores_dict.get(video_name2)
            # print("{}_{}\n".format(video_name1,video_name2))
            pred_rank= 1 if video_score1>=video_score2  else 2
            if sheet_pair=='nonsource':
                gt_labels_non_source.append(video_rank)
                pr_labels_non_source.append(pred_rank)
            elif sheet_pair=='source':
                gt_labels_source.append(video_rank)
                pr_labels_source.append(pred_rank)
    acc_non_source = sum(p == l for p, l in zip(gt_labels_non_source, pr_labels_non_source))/len(gt_labels_non_source)
    acc_source = sum(p == l for p, l in zip(gt_labels_source, pr_labels_source))/len(gt_labels_source)
   
    score=0.45*SROCC+0.45*PLCC+0.05*acc_non_source+0.05*acc_source
    return  score,SROCC,PLCC,acc_non_source,acc_source

import sys
import os
import os.path


# truth_file: the quality label of validation data
# submission_answer_file: the predicted quality label

target_filename = 'prediction.csv'

truth_file = os.path.join(truth_dir, "truth.csv")

score,SROCC,PLCC,acc_non_source,acc_source=metric(submission_answer_file,truth_file,truth_dir)
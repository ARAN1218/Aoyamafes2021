from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def entrance(request):
    return render(request, 'mlapps/entrance.html', {})

questions = {'questions' : {
        'あなたの性別は何ですか？':['0. 男性', '1. 女性']
        ,'あなたは文系ですか？理系ですか？':['0. 文系', '1. 理系']
        ,'あなたの卒業した高校の入学当時の偏差値はおよそいくつでしたか？':['0. ~29', '1. 30~39', '2. 40~49', '3. 50~59', '4. 60~69', '5. 70~']
        ,'あなたは高校時代に何か部活に加入していましたか？':['0. 運動部', '1. 文化部', '2. その他(加入していない含む)']
        ,'あなたは大学進学という進路選択について、明確な目的意識がありましたか？':['0. はい', '1. いいえ']
        ,'あなたはファッションに興味がありますか？':['0. はい', '1. いいえ']
        ,'あなたは周りと合わせるよりも自分の道を突き通す方ですか？':['0. はい', '1. いいえ']
        ,'あなたは音楽を聴きながら勉強していましたか？':['0. はい', '1. いいえ']
        ,'あなたは塾・予備校(またはそれに準ずるサービス等)を利用していましたか？':['0. はい', '1. いいえ']
        ,'あなたは青学合格年度の夏に一日平均どの程度勉強していましたか？':['0. 0~2時間', '1. 2~4時間', '2. 4~6時間', '3. 6~8時間', '4. 8~10時間', '5. 10時間以上']
        ,'あなたは青学合格年度に一日平均どの程度睡眠をとっていましたか？':['0. ~5時間', '1. 6~7時間', '2. 8~時間']
        ,'あなたは受験期にスマホやテレビ等の使用制限をかけていましたか？':['0. はい', '1. いいえ']
        ,'あなたが使用していた付箋の色の数はいくつでしたか？':['0. 0色(付箋は使用していない)', '1. 1色', '2. 2色', '3. 3色', '4. 4色以上']
        ,'あなたが使用していたボールペン(蛍光ペン)の色の数はいくつでしたか？':['0. 0色(ボールペンは使用していない)', '1. 1色', '2. 2色', '3. 3色', '4. 4色以上']
        ,'第一志望大学群はどこでしたか？':['0. 最難関国公立(東京一工)・早慶', '1. 難関国公立(金岡千広等)・上智理科大', '2. 準難関国公立(5S等)・MARCH', '3. 標準国立(stars等)・日東駒専', '4. その他']
        ,'あなたは青学を第何志望校として設定していましたか？':['0. 第一志望校', '1. 第二志望校', '2. 第三志望校', '3. その他(第四志望校以下)']
        ,'夏に何らかの模試を受けましたか？':['0. 河合模試', '1. 駿台模試', '2. 東進模試', '3. 代々木模試', '4. 進研模試', '5. 複数受験した', '6. その他(受けていない含む)']
        ,'[最重要]ハトは"良い"ですか？':['0. はい', '1. いいえ', '810. ｵﾊｰﾄ🐦']
    }}
bayes_columns = ['あなたの性別は何ですか？', 'あなたは文系ですか？理系ですか？', 'あなたは高校時代に何か部活に加入していましたか？', 'あなたは大学進学という進路選択について、明確な目的意識がありましたか？', 'あなたはファッションに興味がありますか？', 'あなたは周りと合わせるよりも自分の道を突き通す方ですか？', 'あなたは音楽を聴きながら勉強していましたか？', 'あなたは塾・予備校(またはそれに準ずるサービス等)を利用していましたか？', 'あなたは青学合格年度に一日平均どの程度睡眠をとっていましたか？', 'あなたは受験期にスマホやテレビ等の使用制限をかけていましたか？', '第一志望大学群はどこでしたか？']
lr_columns = ['あなたは大学進学という進路選択について、明確な目的意識がありましたか？', 'あなたはファッションに興味がありますか？', 'あなたは周りと合わせるよりも自分の道を突き通す方ですか？', 'あなたは音楽を聴きながら勉強していましたか？', 'あなたは塾・予備校(またはそれに準ずるサービス等)を利用していましたか？', 'あなたは受験期にスマホやテレビ等の使用制限をかけていましたか？', '第一志望大学群はどこでしたか？']
svm_columns = ['あなたはファッションに興味がありますか？']
prediction = {
    0.:'A',
    1.:'B',
    2.:'C',
    3.:'D',
    4.:'E'
}
def delete_columns(df, columns):
    for column in df.columns:
        if column not in columns:
            del df[column]

def score(request):
    if request.method == 'GET':
        return render(request, 'mlapps/score.html', questions)
    else:
        
            df_try = pd.DataFrame(index=['own'])
            for question in questions['questions']:
                df_try[question] = request.POST[question]

            df_bayes, df_lr, df_svm = df_try.copy(), df_try.copy(), df_try.copy()
            delete_columns(df_bayes, bayes_columns)
            delete_columns(df_lr, lr_columns)
            delete_columns(df_svm, svm_columns)

            with open('Aoyamasai_models.pickle', mode='rb') as fp:
                model1, model2, model3 = pickle.load(fp)
            pred = int(np.round((model1.predict(df_bayes) + model2.predict(df_lr) + model3.predict(df_svm)) / 3))

            return render(request, 'mlapps/score.html',
            {
                'pred':prediction[pred],
            }
            )
        
            return render(request, 'mlapps/score.html', questions)

def rent(request):
    return render(request, 'mlapps/rent.html', {})

def travel(request):
    return render(request, 'mlapps/travel.html', {})
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
warnings.simplefilter('ignore')
from sklearn.linear_model import LinearRegression as LR
import optuna.integration.lightgbm as lgbm_opt
from matplotlib.ticker import MaxNLocator
import lightgbm as lgbm
import random
from scipy.optimize import minimize
import category_encoders as ce
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix
import pickle


class CFG:
    debug = False
    num_boost = 10000
    num_early = 30
    target_enc = False
    do_learn = False
    learn_optuna = True
    learn_local = False

#  Metric
def smape(y_true, y_pred):
    y_true_plus_1 = y_true + 1
    y_pred_plus_1 = y_pred + 1
    metric = np.zeros(len(y_true_plus_1))
    
    numerator = np.abs(y_true_plus_1 - y_pred_plus_1)
    denominator = ((np.abs(y_true_plus_1) + np.abs(y_pred_plus_1)) / 2)
    
    mask_not_zeros = (y_true_plus_1 != 0) | (y_pred_plus_1 != 0)
    metric[mask_not_zeros] = numerator[mask_not_zeros] / denominator[mask_not_zeros]
    
    return 100 * np.nanmean(metric)



def feval_smape(y_pred, lgb_train):
    
    y_true = lgb_train.get_label()
    y_true_plus_1 = y_true + 1
    y_pred_plus_1 = y_pred + 1
    metric = np.zeros(len(y_true_plus_1))
    
    numerator = np.abs(y_true_plus_1 - y_pred_plus_1)
    denominator = ((np.abs(y_true_plus_1) + np.abs(y_pred_plus_1)) / 2)
    
    mask_not_zeros = (y_true_plus_1 != 0) | (y_pred_plus_1 != 0)
    metric[mask_not_zeros] = numerator[mask_not_zeros] / denominator[mask_not_zeros]
    
    metrics = 100 * np.nanmean(metric)
    
    return 'smape', metrics, False





def enc_vis_month_merge(data, enc_tar_visit_month):
    
    enc_data = data.copy()
    
    enc_data = pd.merge(
        data,
        enc_tar_visit_month,
        on = 'visit_month',
        how = 'left'
    )
    
    return enc_data

def piv_merge(data, peptides, proteins):
    
    pivot_peptides = pd.pivot_table(
        peptides,
        index = ['visit_id', 'visit_month', 'patient_id'], 
        columns = 'Peptide',
        values = 'PeptideAbundance',
        aggfunc = np.max
    ).reset_index(drop = False)
    

    pivot_proteins = pd.pivot_table(
        proteins,
        index = ['visit_id', 'visit_month', 'patient_id'], 
        columns = 'UniProt',
        values = 'NPX'
    ).reset_index(drop = False)
    
    
    grouped_peptides = peptides.groupby(
        [
            'visit_id', 
            'visit_month', 
            'patient_id'
        ]
    )['PeptideAbundance'].describe().reset_index(drop = False)
    
    for col_nm in grouped_peptides.columns:
        if col_nm not in ['visit_id', 'visit_month', 'patient_id']:
            grouped_peptides = grouped_peptides.rename(columns = {col_nm:'gr_peptides_' + col_nm})
    
    
    grouped_proteins = proteins.groupby(
        [
            'visit_id', 
            'visit_month', 
            'patient_id'
        ]
    )['NPX'].describe().reset_index(drop = False)
    
    
    for col_nm in grouped_proteins.columns:
        if col_nm not in ['visit_id', 'visit_month', 'patient_id']:
            grouped_proteins = grouped_proteins.rename(columns = {col_nm:'gr_protein_' + col_nm})
    
    preprocessed_data = pd.merge(
        pivot_proteins,
        data,
        on = ['visit_id', 'visit_month', 'patient_id'],
        how = 'inner'
    )
    
    preprocessed_data= pd.merge(
        preprocessed_data,
        pivot_peptides,
        on = ['visit_id', 'visit_month', 'patient_id'],
        how = 'inner'
    )
    
    preprocessed_data= pd.merge(
        preprocessed_data,
        grouped_peptides,
        on = ['visit_id', 'visit_month', 'patient_id'],
        how = 'inner'
    )
    
    preprocessed_data= pd.merge(
        preprocessed_data,
        grouped_proteins,
        on = ['visit_id', 'visit_month', 'patient_id'],
        how = 'inner'
    )
    
    
    return preprocessed_data


def visualize_importance(models, feat_train_df):
    
    """
    lightGBM の model 配列の feature importance を plot する
    CVごとのブレを box plot として表現します.
    
    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importance()
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
    sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    fig.tight_layout()
    plt.show()
    return fig, ax

def fit_lgbm(x_tra, x_val, y_tra, y_val, params, fold_cnt):

    """
    optunaでチューニングする。

    Args:
        x_tra(DataFrame):学習データの説明変数
        x_val(DataFrame):検証データの説明変数
        y_tra(DataFrame):学習データの目的変数
        y_val(DataFrame):検証データの目的変数
        prams(dict):学習パラメータ
    Returns:
        score(float):学習スコア
        model(model):モデル
        model.params(dict):最適モデルのパラメータ

    """
    
    if CFG.learn_optuna == True:
        lgbm = lgbm_opt
    
    oof_pred = np.zeros(len(y_val), dtype=np.float32)

    trains = lgbm.Dataset(x_tra, y_tra)
    valids = lgbm.Dataset(x_val, y_val)

    if CFG.learn_optuna == True:
        model = lgbm.train(
            params, 
            trains,
            valid_sets = valids,
            num_boost_round = CFG.num_boost,
            verbose_eval = False,
            early_stopping_rounds = CFG.num_early
    )
        
    elif CFG.learn_oputna == False:
        model = lgbm.train(
            params, 
            trains,
            valid_sets = valids,
            num_boost_round = CFG.num_boost,
            verbose_eval = False,
            early_stopping_rounds = CFG.num_early,
            feval = feval_smape
        )

    oof_pred = model.predict(x_val)
    score = smape(y_val, oof_pred)
    
    
    month_smape = pd.DataFrame()
    
    vis_df = x_val.copy()
    vis_df['pred'] = oof_pred
    vis_df['real'] = y_val
    
    for vis_month in vis_df['visit_month'].unique():
        tmp_vis_df = vis_df[vis_df['visit_month'] == vis_month]
        
        month_smape = month_smape.append({
            'visit_month':vis_month,
            'smape_' + str(fold_cnt):smape(tmp_vis_df['real'], tmp_vis_df['pred'])
        }, ignore_index = True)
    

    print('*'*50)
    print('*'*50)
    print('*'*50)
    
    print(model.params)
    print('smape:', score)

    print('*'*50)
    print('*'*50)
    print('*'*50)


    return score, model, model.params, month_smape



def calculate_predictions(pred_month, trend, target):

    if target == 'updrs_4': 
        pred_month = pred_month.clip(60, None)
    
    return np.round(trend[0] + pred_month * trend[1])

def do_infe(data, target, is_rule, models, train, feature_cols, target_to_trend):

    for u in target:
        
        u_num = u.split('_')[1]
        data['result_' + str(u)] = 0

        if is_rule == False:
        
            if CFG.target_enc == True:
                data = enc_vis_month_merge(data, enc_tar_visit_month)
            
            for feature_col in feature_cols:
                if feature_col not in data.columns:
                    data[feature_col] = train[feature_col].mean()        

            X = data.copy()[feature_cols]
            
            if u == 'updrs_4':
                data['result_' + str(u)] = 0
                
            else:
                preds = []
                for model in models[u_num]:

                    sub_pred = model.predict(X.values)
                    preds.append(sub_pred)
                    data['result_' + str(u)] = np.ceil(np.mean(preds, axis = 0))
                    data['result_' + str(u)] = np.where(data['result_' + str(u)] < 0, 0, data['result_' + str(u)])

        else:
            # Predict    
            X = data[['visit_month']]
            
            zero_pred = calculate_predictions(
                    pred_month = np.array([X]),
                    trend=target_to_trend[u],
                    target=u
            )[0]
            data['result_' + str(u)] = zero_pred


    result = pd.DataFrame()

    for m in [0, 6, 12, 24]:
        for u in [1, 2, 3, 4]:
            
            temp = data[["visit_id", "result_updrs_" + str(u)]]
            temp["prediction_id"] = temp["visit_id"] + "_updrs_" + str(u) + "_plus_" + str(m) + "_months"
            temp["rating"] = temp["result_updrs_" + str(u)]      
            result = result.append(temp)            
            
    for m in [0, 6, 12, 24]:
        for u in [1, 2, 3, 4]:
            for visit in result['visit_id'].unique():
                
                visit_month  = int(visit.split('_')[1])
                
                if is_rule == False:
            
                    if u != 1:
                        target_col = 'updrs_' + str(u)
                        prediction_id = visit + "_updrs_" + str(u) + "_plus_" + str(m) + "_months"

                        result.loc[result['prediction_id'] == prediction_id , 'rating'] = calculate_predictions(
                            pred_month = visit_month + np.array([m]),
                            trend=target_to_trend[target_col],
                            target=target_col
                        )[0]
                        
                        
                    if (u == 1) & (visit_month not in [60, 72, 84]):
                        target_col = 'updrs_' + str(u)
                        prediction_id = visit + "_updrs_" + str(u) + "_plus_" + str(m) + "_months"

                        result.loc[result['prediction_id'] == prediction_id , 'rating'] = calculate_predictions(
                            pred_month = visit_month + np.array([m]),
                            trend=target_to_trend[target_col],
                            target=target_col
                        )[0]
                        
                        
                        
                    if (m != 0) & (u == 1):
                        target_col = 'updrs_' + str(u)
                        prediction_zero_id = visit + "_updrs_" + str(u) + "_plus_" + str(0) + "_months"
                        prediction_id = visit + "_updrs_" + str(u) + "_plus_" + str(m) + "_months"
                        zero_value = result.loc[result['prediction_id'] == prediction_zero_id , 'rating']
                        zero_pred = calculate_predictions(
                            pred_month = visit_month + np.array([0]),
                            trend=target_to_trend[target_col],
                            target=target_col
                        )[0]
                        
                        

                        result.loc[result['prediction_id'] == prediction_id , 'rating'] = calculate_predictions(
                            pred_month = visit_month + np.array([m]),
                            trend=target_to_trend[target_col],
                            target=target_col
                        )[0] + zero_value - zero_pred
                    
                        
                else:
                    target_col = 'updrs_' + str(u)
                    prediction_id = visit + "_updrs_" + str(u) + "_plus_" + str(m) + "_months"

                    result.loc[result['prediction_id'] == prediction_id , 'rating'] = calculate_predictions(
                        pred_month = visit_month + np.array([m]),
                        trend=target_to_trend[target_col],
                        target=target_col
                    )[0]
                    
    result = result [['prediction_id', 'rating']]
    result['rating'] = np.where(result['rating'] < 0, 0, result['rating'])
    
    result = result.drop_duplicates(subset=['prediction_id', 'rating']) 
    
    
    return result

def get_predictions(data, peptides,  proteins, sample_submission, models, target, train, feature_cols, target_to_trend):

    all_result = pd.DataFrame()
    
    data_before = data.copy()
    
    data = piv_merge(data, peptides, proteins)
    data = data.drop_duplicates(subset = ['visit_id'])
    data_id = data['visit_id'].unique()  
    data_lr = data_before[~data_before['visit_id'].isin(data_id)]   

    
    result1 = do_infe(
        data = data, 
        target = target, 
        is_rule = False,
        models = models,
        train = train,
        feature_cols = feature_cols,
        target_to_trend = target_to_trend
    )    
        
    
    result2 = do_infe(
        data = data_lr, 
        target = target, 
        is_rule = True,
        models = target_to_trend,        
        train = train,
        feature_cols = feature_cols,
        target_to_trend = target_to_trend

    )    
     
    all_result = pd.concat([result1, result2], axis = 0)

    return all_result

def get_train(CFG):

    if CFG.learn_local == True:
        clinical_path = 'train_clinical_data.csv'
        protein_path = 'train_proteins.csv'
        peptide_path = 'train_peptides.csv'

    else:
        clinical_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv'
        protein_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv'
        peptide_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv'

    target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    clinical_data = pd.read_csv(clinical_path)
    clinical_data["upd23b_clinical_state_on_medication"] = pd.factorize(clinical_data['upd23b_clinical_state_on_medication'])[0]
    print('clinical_read finished')
    print(clinical_data.head())
    print()
    print()


    proteins = pd.read_csv(protein_path)
    print('protein_read finished')
    print(proteins.head())
    print()
    print()

    peptides = pd.read_csv(peptide_path)
    print('peptide_read finished')
    print(peptides.head())
    print()
    print()


    train = piv_merge(clinical_data, peptides, proteins)
    print(train.shape)
    train.head()

    train['null_num'] = train.isnull().sum(axis=1)
    train.head()

    if CFG.target_enc == True:

        for target_nm in target:
            train['visit_month_' + target_nm] =  train['visit_month']
            te = ce.TargetEncoder(cols='visit_month_' + target_nm, handle_unknown='ignore')
            train_te = te.fit_transform(train, train[target_nm])
            train = te.transform(train)
            
        enc_tar_visit_month = train[['visit_month', 'visit_month_updrs_1', 'visit_month_updrs_2', 'visit_month_updrs_3', 'visit_month_updrs_4']].drop_duplicates()
        enc_tar_visit_month

    uniquenum_peptides = peptides.groupby('UniProt')['Peptide'].nunique().reset_index(drop = False)
    uniquenum_peptides.head()

    same_peptides = uniquenum_peptides[uniquenum_peptides['Peptide'] == 1]['UniProt'].unique()
    same_peptides

    remove_cols = peptides[peptides['UniProt'].isin(same_peptides)]['Peptide'].unique()
    remove_cols[:5]

    feature_cols = train.columns
    feature_cols = list(
        set(feature_cols) 
        -set(target) 
        -set(['visit_id', 'patient_id', 'upd23b_clinical_state_on_medication'])
        -set(remove_cols)
    )

    return train, feature_cols

def main(CFG, target_to_torend):
    
    target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    if CFG.learn_local == True:
        clinical_path = 'train_clinical_data.csv'
        protein_path = 'train_proteins.csv'
        peptide_path = 'train_peptides.csv'

    else:
        clinical_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv'
        protein_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv'
        peptide_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv'

    clinical_data = pd.read_csv(clinical_path)
    clinical_data["upd23b_clinical_state_on_medication"] = pd.factorize(clinical_data['upd23b_clinical_state_on_medication'])[0]
    print('clinical_read finished')
    print(clinical_data.head())
    print()
    print()


    proteins = pd.read_csv(protein_path)
    print('protein_read finished')
    print(proteins.head())
    print()
    print()

    peptides = pd.read_csv(peptide_path)
    print('peptide_read finished')
    print(peptides.head())
    print()
    print()


    train = piv_merge(clinical_data, peptides, proteins)
    print(train.shape)
    train.head()

    train['null_num'] = train.isnull().sum(axis=1)
    train.head()

    if CFG.target_enc == True:

        for target_nm in target:
            train['visit_month_' + target_nm] =  train['visit_month']
            te = ce.TargetEncoder(cols='visit_month_' + target_nm, handle_unknown='ignore')
            train_te = te.fit_transform(train, train[target_nm])
            train = te.transform(train)
            
        enc_tar_visit_month = train[['visit_month', 'visit_month_updrs_1', 'visit_month_updrs_2', 'visit_month_updrs_3', 'visit_month_updrs_4']].drop_duplicates()
        enc_tar_visit_month

    uniquenum_peptides = peptides.groupby('UniProt')['Peptide'].nunique().reset_index(drop = False)
    uniquenum_peptides.head()

    same_peptides = uniquenum_peptides[uniquenum_peptides['Peptide'] == 1]['UniProt'].unique()
    same_peptides

    remove_cols = peptides[peptides['UniProt'].isin(same_peptides)]['Peptide'].unique()
    remove_cols[:5]

    feature_cols = train.columns
    feature_cols = list(
        set(feature_cols) 
        -set(target) 
        -set(['visit_id', 'patient_id', 'upd23b_clinical_state_on_medication'])
        -set(remove_cols)
    )

    # model = {}
    models = {
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }

    best_params = {
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }

    scores = {
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }

    pid_train = train['patient_id']
    unique_pid = train['patient_id'].unique()

    for u in target:
            
        all_month_smape = train.copy()
        all_month_smape = train[['visit_month']].drop_duplicates().sort_values('visit_month')
            
        u_num = u.split('_')[1]
        print(u_num)
            
        # Drop NAs
        temp = train.dropna(subset=[u])  
        
        # Train data
        x = temp[feature_cols]
        y = temp[u]
        
        # ----
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_cnt = 0

        # gidごとにグループkfoldを実施する。
        for tr_group_idx, va_group_idx in kf.split(unique_pid):
            
            fold_cnt += 1

            tr_groups, va_groups = unique_pid[tr_group_idx], unique_pid[va_group_idx]

            is_tr = pid_train.isin(tr_groups)
            is_va = pid_train.isin(va_groups)

            x_tra, x_val = x[is_tr], x[is_va]
            y_tra, y_val = y[is_tr], y[is_va]


            if CFG.learn_optuna == True:
                params = {
                    'objective':'regression',
                    'metric': 'rmse'
                }
                
            
            elif CFG.learn_optuna == False:
            
                params = {
                    'objective':'regression',
                    'metric': None
                }
                
            score, model, best_param, month_smape = fit_lgbm(x_tra, x_val, y_tra, y_val, params, fold_cnt)
            scores[u_num].append(score)
            models[u_num].append(model)
            best_params[u_num].append(best_param)
            
            all_month_smape = pd.merge(
                all_month_smape,
                month_smape,
                on = 'visit_month',
                how = 'left'
            )
            
        all_month_smape['smape_mean'] = (all_month_smape['smape_1'] + all_month_smape['smape_2'] + all_month_smape['smape_3'] + all_month_smape['smape_4'] + all_month_smape['smape_5']) / 5
            
        print(all_month_smape)
        # visualize_importance(models[u_num], x)
        
        
    print(scores)

    for num_i in scores:
        print(num_i)
        print(sum(scores[num_i])/len(scores[num_i]))

    for num_i in best_params:
        print(num_i)
        for best_param in best_params[num_i]:
            print(best_param)
            print('*'*80)

    for num_i in models:
        cnt = 0
        print(num_i)
        for model in models[num_i]:
            cnt += 1
            filename = f'updrs_{num_i}_fold{cnt}.pkl'
            with open(filename, mode='wb') as f:
                pickle.dump(model, f)

    return models

def prediction(CFG, test, test_peptides, test_proteins, sample_submission, models, target, target_to_trend):

    train, feature_cols = get_train(CFG)
    result = get_predictions(test, test_peptides, test_proteins, sample_submission, models, target, train, feature_cols, target_to_trend)

    return result

if __name__ == '__main__':

    CFG.learn_local = True
    target_to_trend =     {
        'updrs_1': [5.394793062665313, 0.027091086167821344],
        'updrs_2': [5.469498130092747, 0.02824188329658148],
        'updrs_3': [21.182145576879183, 0.08897763331790556],
        'updrs_4': [-4.645638591439615, 0.08511556737899556]                  
    }
    models = main(CFG, target_to_trend)



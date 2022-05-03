                                                                                           #!/usr/bin/env python
# coding: utf-8
#Dev par Victor Lieutaud
#Ne pas reproduire sans citer la source :*
# In[1]:

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# In[2]:

#fonction agrégation DF
#prend en compte un df principal, un dictionnaire de df "secondaires", la colonne date sur laquel on merge, renvoie le df concaténé
#du genre :
#"Date (CEST)"
def agreg_df(df_principal, dict_df_snd, col_date):
    for key, value in zip(dict_df_snd.keys(), dict_df_snd.values()):
        df_principal= df_principal.merge(value, on = col_date, how = "left")
    return df_principal
# In[3]:
#fonction remplacement NA
#prend en argument un df, éventuellement une méthode, clean, et renvoie le df
def replace_na(df_principal):
    df_principal.fillna(df_principal.median(), inplace = True)
    return df_principal

#%% In[4]
#On crée la fonction pour tester les corr respectives pour modèle VAR
def grangers_test(data, maxlag, variables, test='ssr_chi2test',verbose=False):    
    """Les valeurs dans le df sont les p-valeurs
    L'hypothèse H0 de notre test est la suivant :
        "Les prédictions de la série X n'influence pas les prédictions de la série Y"
    Ce qui signifie qu'une p-valeur inférieure à 0.05 rejette l'hypothèse H0 et incite à garder ce couple de valeurs
    Comme on s'intéresse à la prédiciton de la variable 1, on ne va jamais l'abandonner

Les arguments sont :
    Data, le DF de nos valeurs
    maxlag, le fameux maxlag pour le nombre de paramètres dans l'équation'
    variables : une list qui contient le nom des variables c'est à dire le nom de nos colonnes'
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in df.columns:
        for row in df.index:
            test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)] #on va avoir toutes les p-valeurs une part lag
            min_p_value = np.min(p_values) #On s'intéresse à la valeur minimale des p-valeur
            df.loc[row, col] = min_p_value
    df.columns = [var + '_X' for var in variables]
    df.index = [var + '_Y' for var in variables]
    return df

#Fonction test stationarité
def dickey_fuller_test(data, seuil_signif=0.05, name='', verbose=False):
    """On conduit un test de Dick-Fuller sur notre data set et on imprime les résultats"""
    """On rappelle que pour rejeter l'hypothèse H0 il faut que la p-value soit inférieur au seuil choisi, ici 0.05"""
    result = adfuller(data, autolag='AIC')
    output = {'statistique_du_test':round(result[0], 4), 'p_value':round(result[1], 4), 'n_lags':round(result[2], 4), 'n_obs':result[3]}
    p_value = output['p_value'] 

    # Print Summary
    print(f'    Test de Dick-Fuller augmenté sur : "{name}"', "\n   ", '~'*47)
    print(f' Hypothèse H0: La data présente une racine unitaire, la série est Non-Stationnaire.')
    print(f' La p-value est de      : {p_value}')
    print(f' Niveau de confiance    = {seuil_signif}')
    print(f' Statistique de test    = {output["statistique_du_test"]}')
    print(f' Nombre de lags choisis = {output["n_lags"]}')

    if p_value <= seuil_signif:
        print(f" => P-Value = {p_value} <= {seuil_signif}. On rejette H0 au seuil de confiance.")
        print(f" => La série est STATIONNAIRE.")
    else:
        print(f" => P-Value = {p_value} > {seuil_signif}. On ne peut pas rejeter H0 au seuil de confiance.")
        print(f" => La série est NON stationnaire.")
        
# In[5]:
#Fonction metric cible R^2 ajusted
#Prend en argument le r2, et le DF qui sert au test pour le nb de lignes et col
def r2_ajusted(r2, df):
    longueur = len(df.index)
    nb_col = len(df.columns)
    score_ajusted = 1-((1-r2)*(longueur-1) / (longueur - nb_col -1))
    return score_ajusted

# In[6]:
#Fonction plotting historique, True, Pred
def plot_pred_true(y_pred, y_true):
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15,10), dpi = 80)
    test_pred, = ax.plot(date_du_split, y_pred, color = "b")
    test_test, = ax.plot(date_du_split, y_true, color = "g")
    ax.legend((test_pred, test_test), ("prediction", "réel"), loc = "upper left")
    fig.show()
    return plt.savefig("resultat.png")

#Fonction plot historique total avec pred
def plot_pred_true_total(y_pred, y_true):
    date_list = df.index.tolist() + date_du_split.tolist()
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15,10), dpi = 80)
    test_pred, = ax.plot(date_du_split, y_pred, color = "b")
    test_test, = ax.plot(date_totale, variable_cible, color = "g")
    ax.legend((test_pred, test_test), ("prediction", "réel"), loc = "upper left")
    fig.show()
    return plt.savefig("Historique et true VS prédiction.png")

# In[7]:
#pour detransfo data :
def inv_diff(df_train, df_forecast, n_jour_cible, second_diff = False):
    df_fc_inv = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        """
        Attention ! lorsque l'on inverse une différenciation, un récupère la "dernière donnée" à laquelle on rajoute
        la somme cumulée des valeurs différenciées.
        Dans le cas d'un array prédit, le dernier point de donnée n'est PAS le "-1" du train_set, mais le "-1-nb_jour_pred"
        Dans notre cas on va donc retourner 120 jours en arrière
        Dans le cas d'une double diff, on soustrait d'abord l'avant dernière valeur à la dernière
        """
        if second_diff:
            df_fc_inv[str(col)+'_1d'] = (df_train[col].iloc[-n_jour_cible-1]-df_train[col].iloc[-n_jour_cible-2]) + df_fc_inv[str(col)].cumsum()
            df_fc_inv[str(col)+'_forecast'] = df_train[col].iloc[-n_jour_cible-1] + df_fc_inv[str(col)+"_1d"].cumsum()
        else:
            df_fc_inv[str(col)+'_forecast'] = df_train[col].iloc[-n_jour_cible-1] + df_fc_inv[str(col)].cumsum()
    return df_fc_inv

#On va créer la fonction pour fit et evaluer le modèle var
def pipeline_var(model, x, y, lag, df_orig, n_jour_cible):
    model_fitted_var = model.fit(lag, ic = "aic", trend = "ct")
    result_normality = model_fitted_var.test_normality().pvalue #"H0 : les données suivent une loi normale. Si p-value <0.05, on rejette
    result_whiteness = model_fitted_var.test_whiteness(round((len(x) + len(y))/5)).pvalue 
    #On choisit dans la formule précédente l = nb_observ/5
    #On calcule la moyenne des résidus (biais) que l'on va ajouter à nos preds
    df_resid = model_fitted_var.resid
    mean_resid = df_resid.mean()

    lag_order = model_fitted_var.k_ar
    input_data = x.values[-lag_order:]
    y_predicted = model_fitted_var.forecast(y = input_data, steps = len(y))
    y_predicted = y_predicted + mean_resid[0] #j'ajoute le biais
    
    """Pour le moment, nos résidus ne passent pas les tests de normalité et de bruits blancs
    Je ne peux donc pas réaliser d'intervalles de confiance classiques
    """
    #forecast_interval = model_fitted_var.forecast_interval(y = input_data, steps = len(y), alpha = 0.05)
    #df_interval_low = pd.DataFrame.from_records(forecast_interval[0] - forecast_interval[1], columns = df_diff_2.columns)
    #df_interval_up = pd.DataFrame.from_records(forecast_interval[0] + forecast_interval[2], columns = df_diff_2.columns)
        
    #il faudrait coder ici du conditionnel en cas de changement de metric
    df_predicted = pd.DataFrame(y_predicted, index=y.index, columns=x.columns)
    
    df_true_results = inv_diff(df_orig, df_predicted, n_jour_cible)
    #df_interval_low = inv_diff(df_orig, df_interval_low, n_jour_cible)
    #df_interval_up = inv_diff(df_orig, df_interval_up, n_jour_cible)
    y_test_predicted = (df_true_results["Variable 1_forecast"].apply(lambda x: inv_boxcox(x,fitted_lambda[0])))
    #Sur la ligne précédente, on va détransformer la transformation cox-box en appliquant "inv boxcox" avec en paramètre le lambda associé
    #y_test_predicted_low = (df_interval_low["Spot PEG DA_forecast"].apply(lambda x: inv_boxcox(x,fitted_lambda[0])))
    #y_test_predicted_up = (df_interval_up["Spot PEG DA_forecast"].apply(lambda x: inv_boxcox(x,fitted_lambda[0])))
    r2_var = r2_ajusted(r2_score(test_labels, y_test_predicted), x)
    MAE_var = mean_absolute_error(test_labels, y_test_predicted)
    return y_test_predicted, MAE_var, r2_var, result_normality, result_whiteness


# In[8]
#Données "Perso", attention ce n'est pas diffusable !
df = pd.read_excel(r"C:\Users\victo\Desktop\data VAR.xlsx")

nom_col_date_df = "Date (CEST)"
df = df.set_index(nom_col_date_df)

nom_col = []
for i in range(1,29):
    nom_col.append("Variable " + str(i))
nom_col_actu = []
for col in df.columns:
    nom_col_actu.append(col)

zip_ite = zip(nom_col_actu, nom_col)
dict_col = dict(zip_ite)

df.rename(columns = dict_col, inplace = True)

# In[9]:
#On définit la variable cible du modèle
variable_cible = df["Variable 1"]
#On récupère l'index du DF qui va nous servir plus tard pour le plotting :)
date_totale = df.index
# In[31]:
#Remplacement NA
replace_na(df)

#%% In[10]:
"""On choisit ici le nombre de jour cible pour notre modèle, c'est à dire le nombre de jour prédit
"""
n_jour_cible = 120

#%% In[╬11]:
#On crée nos DF pour VAR selon la règle du 4/5 1/5  et de la fenêtre à Xjour jours
df_pour_var = df.iloc[-5*n_jour_cible:, :]
test_labels = df_pour_var.iloc[-120:, 0]
date_du_split = date_totale[-n_jour_cible:]

#Je rajoute ici "1", car parfois la data est égale à 0 pile (time series) et cela fait 
df_pour_var = df_pour_var +1

#%% In[12]: 
#transfo box-cox 
df_pour_var_transfo, fitted_lambda = df_pour_var.apply(lambda x: boxcox(x, lmbda = None)[0]), df_pour_var.apply(lambda x: boxcox(x, lmbda = None)[1])

#%% In[13]:
#On teste la stationarity de nos var 
#On teste ici la stationarité du df
for name, column in df_pour_var_transfo.iteritems():
    dickey_fuller_test(column, name=column.name)
    print('\n')
    
#Il faut faire au moins une diff :
df_diff_1 = df_pour_var_transfo.diff().dropna()
#On rerun le test
for name, column in df_diff_1.iteritems():
    dickey_fuller_test(column, name=column.name)
    print('\n')

#Après 1 diff, la série n'est pas totalement diff, donc on recommence

df_diff_2 = df_diff_1.diff().dropna()
for name, column in df_diff_2.iteritems():
    dickey_fuller_test(column, name=column.name)
    print('\n')
#Après 2diff, Ok! Mais deux familles de modèle

#%% In[14]
#création test/train
var_a_drop = ["Variable 5", "Variable 3"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

#%% In[15]
#On va tester ici les différentes corr pour notre modèle var
#On def le maxlag au maximum
maxlag=67
#on fit le modèle sur la data transformée
model_var = VAR(endog = train_var)
res = model_var.select_order(maxlags=maxlag)

#Premier DF Granger
df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 8"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)
#Pour les drops, le tableau se lit de la manière suivante : si valeur <0.05 alors la variable X détermine la variable Y
#on drop ce qui n'est pas corr avec spot peg DA

var_a_drop = ["Variable 7"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 6"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 4"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 16"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 11"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 2"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 15"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 18"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 9"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 20"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 17"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 21"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)


var_a_drop = ["Variable 22"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)


var_a_drop = ["Variable 10"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 19"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

var_a_drop = ["Variable 14"]
df_diff_1 = df_diff_1.drop(var_a_drop, axis = 1)
df_pour_var_transfo = df_pour_var_transfo.drop(var_a_drop, axis = 1)
train_var = df_diff_1[:-n_jour_cible]
test_var = df_diff_1[-n_jour_cible:]

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

#Test Git 2ieme
# In[16]: On va tester ici tous les lags jusqu'au lag max pour l'opti de notre modèle var : peu de temps de calcul super rentable
lag = maxlag #On choisi le lag max
array_mae_var = []
array_pvalue_normal = []
array_pvalue_whiteness = []
for i in range(1, lag):
    array_predit_var, var_MAE, r2_var, pvalue_normality, pvalue_whiteness = pipeline_var(model_var, train_var, test_var, i, df_pour_var_transfo, n_jour_cible)
    array_mae_var.append(var_MAE)
    array_pvalue_normal.append(pvalue_normality)
    array_pvalue_whiteness.append(pvalue_whiteness)
    print(i)
 #↔Dans notre cas et avant bug, le meilleur lag est de 9
#Je regarde mes courbes et je prends aussi l'index+1 de la meilleure MAE

maxlag=44
#on fit le modèle sur la data transformée
model_var = VAR(endog = train_var)
res = model_var.select_order(maxlags=maxlag)

df_granger = grangers_test(df_diff_1, maxlag, variables = df_diff_1.columns)

lag_final = 44

array_predit_var, var_MAE, r2_var, pvalue_normality, pvalue_whiteness = pipeline_var(model_var, train_var, test_var, lag_final, df_pour_var_transfo, n_jour_cible)
plot_pred_true(array_predit_var.values,test_labels)
plot_pred_true_total(array_predit_var.values, test_labels)
print("La MAE du meilleur modèle est de ", var_MAE)
print("Le R2 ajusté du meilleur modèle est de ", r2_var)
print("Le test de blancheur des résidues renvoie une p-valeur de ", round(pvalue_whiteness, 4))
print("le test de normalité des résidus renvoie un pvaleur de ", pvalue_normality)


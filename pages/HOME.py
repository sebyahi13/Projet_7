import streamlit as st 
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik
import time
from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split, validation_curve, learning_curve, cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix,precision_score,recall_score



import warnings
warnings.filterwarnings('ignore')

def plot_roc_curve(fper, tper):
    figure_00=plt.figure(figsize=(12,6))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    #plt.show()    
    st.pyplot(figure_00)






st.title('Implémentez un modèle de scoring')

st.write('le projet 7  ')



seuils=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#couts_seuils=test_222[['cout_seuil_01','cout_seuil_02','cout_seuil_03',
#                        'cout_seuil_04','cout_seuil_05','cout_seuil_06',
#                        'cout_seuil_07','cout_seuil_08','cout_seuil_09','cout_seuil_10']].sum()  

figure_net=plt.figure(figsize=(12,6))
fig, ax = plt.subplots(figsize=(12, 6))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1 ))
plt.grid(visible=True) 
plt.title(' Cout metier bank en Fonction du seuil d acceptation (entre 0.1 et 1.0 )', fontsize=15)
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])

plt.xlabel("seuil", fontsize=20) 
plt.ylabel("cout ", fontsize=20)  
plt.plot(seuils,seuils)

st.pyplot(fig)


from sklearn.metrics import roc_curve

# Receiver Operating Characteristic Curve

fper, tper, thresholds = roc_curve([0,1,1,0,1,1,0], [0,1,1,0,1,1,0])
plot_roc_curve(fper, tper)






fig2=plt.figure(figsize=(6,6))
#ari=metrics.adjusted_rand_score(df_list_xception_1[2]['label_xception'], df_list_xception_1[2]['product_category1_num'])
plt.title("   : " ,fontsize=18 )
 

plt.yticks(fontsize=12,fontweight='bold')
plt.xticks(rotation=90,fontsize=12,fontweight='bold')
sns.heatmap(confusion_matrix([0,1,1,0,1,1,0],[0,1,1,0,1,1,0] ) ,
            annot=True,
            linewidth=.5,    
            fmt='d',
            cbar=False,
            annot_kws={"size": 20},
            cmap="crest" )   
plt.ylabel('true',fontsize=18)
plt.xlabel('pred',fontsize=18);

st.pyplot(fig2)




select_id = st.selectbox(
    'Selectionner un ID Client',
    (-1,4125, 4577, 4589))
if (select_id!=-1):
    st.write('vous avez selectionné:', select_id)
    
    




    
fig3=plt.figure(figsize=(6,6))

a=[0,1,1,0,1,1,0,1,4,4,0,1] 
plt.hist(a,bins=4)

st.pyplot(fig3)

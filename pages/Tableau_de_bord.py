import streamlit as st 
import  matplotlib.pyplot as plt
import  pandas as pd

import time
import joblib
import shap
import plotly.graph_objects as go


html_txt = """
    <div style="background-color: Cyan; padding:5px; border-radius:8px">
    <h1 style="color: black; text-align:center">HOME CREDIT DEFAULT RISK</h1>
    </div>    
    """
st.markdown(html_txt, unsafe_allow_html=True)

st.sidebar.title("Pret a depenser")

df_test_pkl = pd.read_pickle("df_test.pkl")  
df_test_X=df_test_pkl.drop(columns=['SK_ID_CURR','TARGET'])
id_clients=df_test_pkl.SK_ID_CURR.values


#st.header('Score du Client ')
st.markdown('**Score du Client :**')

@st.cache(show_spinner=False, suppress_st_warning=True,allow_output_mutation=True) 
def load_model():
    model = joblib.load('best_model_ok.pkl') 
    explainer = shap.TreeExplainer(model)  
    #shap_values =explainer.shap_values(X)
    return model,explainer

model_ok,explainer_ok= load_model()


#st.write(df_test_pkl)
id_client = st.sidebar.selectbox('Choisir un Identifiant client: ', id_clients)
if (id_client==30750711):
    score=0.8
    st.empty()
else:
    score=0.5
    st.empty()

X_client=df_test_pkl[df_test_pkl['SK_ID_CURR']==id_client]

X_client1=X_client.drop(columns=['SK_ID_CURR','TARGET'])
#st.write(X_client1)

X_client2=X_client1.values

y_pred_proba_model= model_ok.predict_proba(X_client2 )
#st.write(y_pred_proba_model[0][0])
score=y_pred_proba_model[0][0]
#st.empty()
col1,col2=st.columns(2)


with col1:
    st.markdown('<span  style=\"margin:8px;background:green;\"  >', unsafe_allow_html=True  )

    if (score > 0.5):
        st.image('OK.jpg',caption='Decision',width=200)
    else:
        st.image('NOK.png',caption='Decision',width=200)

    st.markdown('</span>', unsafe_allow_html=True  )

with col2:

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score * 100,
        mode = "number+gauge",
        title = {'text': "Score du Client : (0-100)"},
        delta = {'reference': 0.5},
        gauge = {'axis': {'range': [0, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "red"},
                 {'range': [50, 100], 'color': "MediumSpringGreen"}],
             'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 0.80, 'value': round(score * 100,4)     }}))



    fig.update_layout(
    autosize=False,
    width=500,
    height=300,
    margin=dict(
        l=10,
        r=10,
        b=50,
        t=50,
        pad=4
    ),
    paper_bgcolor="white",
    )
    st.plotly_chart(fig)



#st.write(X_client)
#st.write(X_client1)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
X_client1 = X_client1.reset_index(drop=True)
#st.header('Donnees Client')
st.markdown('**Donnees Client :**')

#st.title('title')
infos_client = pd.DataFrame({})
#Sexe du client
#CODE_GENDER

code_gender={0:'FEMME',1:'HOMME'     }
#st.write(   code_gender[ 0  ]     )
#X_client1['CODE_GENDER'].values
#st.write(    X_client1.loc[0, 'CODE_GENDER']      )
infos_client = infos_client.append({
    'INFORMATION' : 'Sexe du client' ,
    'VALEUR' :    code_gender[   X_client1.loc[0, 'CODE_GENDER']    ]  ,  
     
    },ignore_index=True )




flag_own_car={0:'NON',1:'OUI'     }
infos_client = infos_client.append({
    'INFORMATION' : 'Possession (Voiture)' ,
    'VALEUR' :    flag_own_car[   X_client1.loc[0, 'FLAG_OWN_CAR']    ]  ,  
     
    },ignore_index=True )




flag_own_realty={0:'NON',1:'OUI'     }
infos_client = infos_client.append({
    'INFORMATION' : 'Possession (Villa/App)' ,
    'VALEUR' :    flag_own_realty[   X_client1.loc[0, 'FLAG_OWN_REALTY']    ]  ,  
     
    },ignore_index=True )




infos_client = infos_client.append({
    'INFORMATION' : 'Nombre (Enfants)' ,
    'VALEUR' :       X_client1.loc[0, 'CNT_CHILDREN']     ,  
     
    },ignore_index=True )





infos_client = infos_client.append({
    'INFORMATION' : 'Revenu' ,
    'VALEUR' :       X_client1.loc[0, 'AMT_INCOME_TOTAL']     ,  
     
    },ignore_index=True )




st.write(infos_client.T)



infos_client1 = pd.DataFrame({})



infos_client1 = infos_client1.append({
    'INFORMATION' : 'Montant (Credit)' ,
    'VALEUR' :       X_client1.loc[0, 'AMT_CREDIT']     ,  
     
    },ignore_index=True )



infos_client1 = infos_client1.append({
    'INFORMATION' : 'Annuit?? (Pret)' ,
    'VALEUR' :       X_client1.loc[0, 'AMT_ANNUITY']     ,  
     
    },ignore_index=True )

#NAME_EDUCATION_TYPE_Academic degree
#NAME_EDUCATION_TYPE_Secondary / secondary special
#NAME_EDUCATION_TYPE_Higher education
#NAME_EDUCATION_TYPE_Incomplete higher
#NAME_EDUCATION_TYPE_Lower secondary
name_education_type="Unknown"
if (  X_client1.loc[0, 'NAME_EDUCATION_TYPE_Lower secondary'] ==1  ):
    name_education_type="Lower secondary"

if (  X_client1.loc[0, 'NAME_EDUCATION_TYPE_Academic degree'] ==1  ):
    name_education_type="Academic degree"

if (  X_client1.loc[0, 'NAME_EDUCATION_TYPE_Secondary / secondary special'] ==1  ):
    name_education_type="Secondary / secondary special"

if (  X_client1.loc[0, 'NAME_EDUCATION_TYPE_Incomplete higher'] ==1  ):
    name_education_type="Incomplete higher"

if (  X_client1.loc[0, 'NAME_EDUCATION_TYPE_Higher education'] ==1  ):
    name_education_type="Higher education"




infos_client1 = infos_client1.append({
    'INFORMATION' : 'Niveau (Etudes)' ,
    'VALEUR' :       name_education_type    ,  
     
    },ignore_index=True )


# NAME_FAMILY_STATUS_Single / not married
# NAME_FAMILY_STATUS_Married
# NAME_FAMILY_STATUS_Civil marriage
# NAME_FAMILY_STATUS_Widow
# NAME_FAMILY_STATUS_Separated



name_family_status="Unknown"
if (  X_client1.loc[0, 'NAME_FAMILY_STATUS_Single / not married'] ==1  ):
    name_family_status="Single / not married"

if (  X_client1.loc[0, 'NAME_FAMILY_STATUS_Married'] ==1  ):
    name_family_status="Married"

if (  X_client1.loc[0, 'NAME_FAMILY_STATUS_Civil marriage'] ==1  ):
    name_family_status="Civil marriage"

if (  X_client1.loc[0, 'NAME_FAMILY_STATUS_Widow'] ==1  ):
    name_family_status="Widow"

if (  X_client1.loc[0, 'NAME_FAMILY_STATUS_Separated'] ==1  ):
    name_family_status="Separated"


infos_client1 = infos_client1.append({
    'INFORMATION' : 'Situation (Familiale)' ,
    'VALEUR' :    name_family_status    ,  
     
    },ignore_index=True )

age=  ( (-1) * X_client1.loc[0, 'DAYS_BIRTH'] ) / 365
age=  round(age ,1 ) 
infos_client1 = infos_client1.append({
    'INFORMATION' : 'Age du client' ,
    'VALEUR' :  age  ,  
     
    },ignore_index=True )




st.write(infos_client1.T)





# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#st.header('Variables Importantes ')
st.markdown('**Variables Importantes :**')

infos_variables= pd.DataFrame({})


infos_variables = infos_variables.append({
    'VARIABLE' : 'EXT_SOURCE_1' ,
    'DEFINITION' :  'Score normalis?? de la source (1) de donn??es externe'  ,  
     
    },ignore_index=True )

infos_variables = infos_variables.append({
    'VARIABLE' : 'EXT_SOURCE_2' ,
    'DEFINITION' :  'Score normalis?? de la source (2) de donn??es externe'  ,  
     
    },ignore_index=True )

infos_variables = infos_variables.append({
    'VARIABLE' : 'EXT_SOURCE_3' ,
    'DEFINITION' :  'Score normalis?? de la source (3) de donn??es externe'  ,  
     
    },ignore_index=True )

st.write(infos_variables)








# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#st.columns.
#go.Indicator()



fig1, ax1 = plt.subplots(figsize=(10, 10))
#explainer = shap.TreeExplainer(model)
shap_values1 = explainer_ok.shap_values( X_client1 )
shap.summary_plot(shap_values1, features= X_client1  , plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
st.pyplot(fig1)  



#fig2, ax2 = plt.subplots(figsize=(10, 10))
#explainer = shap.TreeExplainer(model)
#shap_values1 = explainer_ok.shap_values( X_client )
#shap.summary_plot(shap_values1, X_client, feature_names=X_client.columns, show=False, plot_size=None)
#st.pyplot(fig2)  

fig3 ,ax_3= plt.subplots(figsize=(6,6))
ax_3= shap.plots._waterfall.waterfall_legacy(explainer_ok.expected_value[1],shap_values1[1][0], 
                                             feature_names = X_client1.columns,max_display = 20)
st.pyplot(fig3)






#time.sleep(2.4)
#score=0.8
#st.empty()


import streamlit as st 
import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score
from matplotlib.colors import ListedColormap 
import warnings
warnings.filterwarnings('ignore')

st.title('Logistic Regression Viewer By Aditya')

st.subheader('Enter Different Datasets, get Logistic regression results')

dataset_name = 'Example'

uploaded_file = st.file_uploader("Choose a file",type=["csv"])
X = []
y = []

def get_dataset():

    if uploaded_file is not None:

        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        target = str(st.text_area("Enter the Target Column: "))
        required = str(st.text_area("Enter the Required Columns: "))

        required_columns = required.split(",")

        if len(required_columns)!= 0:
            X = dataframe[required_columns]
        y = dataframe[target]
    
        return X, y

if __name__ == "__main__":

    try:
        X,y = get_dataset()
    except:
        print("ERROR IN GETTING DATASET")
        #======================================================Training=========================================================
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        model = LogisticRegression(random_state=0, max_iter=2000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test,y_pred)
        rec = recall_score(y_test,y_pred)

        st.write(f'Accuracy =', acc)
        st.write(f'Precision =', pre)
        st.write(f'Recall =', rec)

    except:
        print("ERROR IN MODEL")
        #==============================================AUC Curve Plot======================================
    try:
        fig = plt.figure()

        y_pred_proba = model.predict_proba(X_test)[::,1]

        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        plt.plot(fpr, tpr, label="AUC="+str(auc))

        plt.legend(loc=4)

        st.pyplot(fig)

        fig1 = plt.figure()

        #===============================================Confusion Matrix Plot==========================================
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

        class_names=[0,1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot(plt.show())
    except:
        print("ERROR IN PLOTTING")
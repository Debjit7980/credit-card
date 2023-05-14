import pandas as pd
import seaborn as sns
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import  confusion_matrix,classification_report
#from streamlit.uploaded_file_manager import UploadedFile

st.title("Credit Card Fraud Detection\n")
st.write("")
st.write("")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
st.write("")
st.write("")
if uploaded_file is not None:
    try:
        
        df=pd.read_csv(uploaded_file)
        st.session_state.df=df
        st.session_state.uploaded_file=uploaded_file
        #df=df.sample(frac=0.1,random_state=48)
        if uploaded_file:
            st.write("The first few records of the dataset:")
            st.write(df.head(70))
            if st.sidebar.checkbox("Show the DataSet"):
                st.write("The Sample of the dataset is: ")
                st.write("")
                st.write(df.head(30))
                st.write("")
                st.write("")
                st.write("Shape of the dataset: \n",df.shape)
                st.write("Data Description: \n",df.describe())
            st.write("")
            st.write("")
            a = int(df["Class"].value_counts()[0])
            b = int(df["Class"].value_counts()[1])
            perc_fra=b/(a+b)*100
            if st.sidebar.checkbox("Show the Genuine and Fraud Transaction Details"):
                st.write(f"Fraudulent Transactions are: {round(perc_fra,3)}%")
                st.write(f"Fraud Cases: {b}")
                st.write(f"Genuine Cases: {a}")
            st.sidebar.write("To See the Genuine and Fraud cases of the dataset in form of charts")
            chart_style=st.sidebar.selectbox(
                label="Select the type of chart",
                options=["Scatter Plot","Histogram","Pie Chart"]
            )
            
            st.write("")
            st.write("")
            st.write("Distribution of Genuine and Fraud Transactions of various charts")
            st.write("")
            st.write("")
 
            if chart_style=="Histogram":

                fig, ax = plt.subplots(figsize=(8,6))

                # Create a bar chart with the values of a and b
                values = [a, b]
                labels = [f"Genuine {a}", f"Fraud {b}"]
                width=0.3
                colors=['#FF7F50', 'red']
                ax.bar(labels, values,width=width,color=colors,)
                ax.set_ylabel('Values')
                ax.set_title('Hisogram')
                st.pyplot(fig) 
                # Show the chart in Streamlit

            if chart_style=="Pie Chart":    
                values=[a,b]
                labels=[f"Genuine\n{a}",f"Fraud\n{b}"]
                explode=[0.1,0]
                colors=['#99ff99','red']
                fig1, ax1 = plt.subplots(figsize=(3,4))
                ax1.pie(values,labels=labels,autopct='%1.1f%%',colors=colors,explode=explode)   
                st.pyplot(fig1) 

            if chart_style=="Scatter Plot":
                class_0 = df[df['Class'] == 0]
                class_1 = df[df['Class'] == 1]

                # Create scatter plots
                scatter_plot_0 = alt.Chart(class_0).mark_circle(size=60,color="green").encode(
                    x='Time',
                    y='Amount'
                ).interactive()

                scatter_plot_1 = alt.Chart(class_1).mark_circle(size=60,color="red").encode(
                    x='Time',
                    y='Amount'
                ).interactive()

                # Display scatter plots
                st.write("Scatter plot of Number of Genuine Transaction")
                st.altair_chart(scatter_plot_0, use_container_width=True)

                st.write("Scatter plot of Number of Fraud Transaction")
                st.altair_chart(scatter_plot_1, use_container_width=True)

            X=df.drop(["Class"],axis=1)
            Y=df["Class"]
            st.write("")
            st.write("")
            st.write("")
            #Split the dataset into training and testing sets
            size=st.sidebar.slider('Test Set Size',min_value=0.2,max_value=0.4)
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=size,random_state=42)
            #if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
            st.write("Distribution of data over Training and Test Set")
            st.write("")
            st.write("")
            st.write("X_Train: ",X_train.shape)
            st.write("Y_Train: ",Y_train.shape)
            st.write("X_Test: ",X_test.shape)
            st.write("Y_Test: ",Y_test.shape)

            algo=st.sidebar.selectbox(label="Select a model for training  and predicting",
                                       options=["Logistic Regression","Extra Trees"]
                                       )


            def compute_performance(model,X_train,Y_train,X_test,Y_test,name_algo):
                scores=cross_val_score(model,X_train,Y_train,cv=3,scoring='accuracy').mean()
                model.fit(X_train,Y_train) #fitting the train_set into the model
                y_pred=model.predict(X_test)
                cm=confusion_matrix(Y_test,y_pred)
                conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
                cf=classification_report(Y_test,y_pred,output_dict=True)
                cf_data=pd.DataFrame(cf).transpose()
                st.write("")
                st.write("")
                #HeatMap for Confusion Matrix
                st.write("The Heat Map of the Predicted Result using "+name_algo+ " model: ")
                st.write("")
                st.write("")
                fig,ax=plt.subplots(figsize=(5,4))
                sns.heatmap(cm,annot=True,center=1,annot_kws={"fontsize":8},ax=ax)
                sns.set(font_scale=1.2)
                ax.set_xlabel("Predicted Result",fontsize=10)
                ax.set_ylabel("True Values",fontsize=10)
                ax.xaxis.set_ticklabels(["Genuine","Fraud"],fontsize=9)
                ax.yaxis.set_ticklabels(["Genuine","Fraud"],fontsize=9)
                st.pyplot(fig)
                st.write("")
                st.write("")
                st.write("Confusion Matrix using "+name_algo+" model :", conf_matrix)  #printing the Confusion Matrix
                st.write("")
                st.write("")
                st.write("Classification Report of "+name_algo+" model: ")  
                st.dataframe(cf_data)  #Printing the Classfication Report in form of dataframe of the model
                st.write("")
                st.write("")
                st.write("Accuracy: ",scores)

            if algo=="Logistic Regression":
                model=LogisticRegression()
                name_algo="Logistic Regression"
                compute_performance(model,X_train,Y_train,X_test,Y_test,name_algo)
            if algo=="Extra Trees":
                model=ExtraTreesClassifier()
                name_algo="Extra Trees"
                compute_performance(model,X_train,Y_train,X_test,Y_test,name_algo)

        else:
            st.write("Upload the appropriate csv file first")
    except:
        st.write("Please enter the creditcard.csv file, anyother csv file will not work")





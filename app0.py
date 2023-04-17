import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1)  Come scaricare la cartella creata su GitHub
#           git clone https://github.com/nemesiMark/app.git

# 2)  Come uplodare file su GitHub
#           git add .
#           git commit -m "nome modifica"
#           git push

# 3)  Come runnare su streamlit
#           steamlit run app0.py


def get_df_info(df):

    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().split('\n')

    # st.dataframe(lines[0:3])
    # st.dataframe(lines)

    for x in lines:
        st.text(x)


def main():

    st.title("Data Transformation")
    uploaded_file = st.file_uploader("Choose your file XLSX:")

    if uploaded_file is not None:

        df = pd.DataFrame()
        flag = True

        print(uploaded_file.name[-4:])
        print(type(uploaded_file.name[-4:]))

        if uploaded_file.name[-4:] == "xlsx":

            ##################################### TRANSFORMATION #####################################
            df = pd.read_excel(uploaded_file)

        else:
            st.warning("XLSX file is required.")

        numeric_cols = df.select_dtypes(
            include=['int', 'float']).columns.tolist()
        
        df1 = df.copy()
        df_standardized = df.copy()
        df_standardized[numeric_cols] = StandardScaler(
        ).fit_transform(df[numeric_cols])
        ##################################### SHOW THE DATAFRAME #####################################
        st.header('Dataframe view')
        st.dataframe(df)
        #############################################################################################
        myPCA = 0

        # columns = st.sidebar.multiselect("Enter the columns name to fill NaN with 0", df.columns)

        # st.header('Choose the operation:')

        choose = 0

        
        choose = st.radio("", ["Info", "Describe", "Correlation",
                              "Box Plot", "Histogram", "PCA"], horizontal=True)
        

        if choose == "Info":

            get_df_info(df)

        if choose == "Describe":

            st.text("Information on dataframe:")
            st.dataframe(df.describe().T)

        

        if choose == "Correlation":

            # Riga per visualizzare la tabella numerica di correlazione
            # st.dataframe(df.corr())
            # se la figura dovesse risultare piccola posso aumentare la figsize=(20,10) espressa in pollici
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(df.corr(), annot=True, ax=ax)
            ax.set_title("Heatmap of correlation")
            st.pyplot(fig)

        if choose == "Box Plot":

            fig, ax = plt.subplots(figsize=(20, 10))
            df.boxplot(ax=ax)
            ax.set_title("Box Plot for outlyers")
            st.pyplot(fig)

        if choose == "Histogram":

            fig, ax = plt.subplots(figsize=(20, 10))
            df.hist(ax=ax)
            ax.set_title("Histogram")
            st.pyplot(fig)

        if choose == "PCA":

            myPCA = PCA().fit(df_standardized[numeric_cols])
            fig = plt.figure(figsize=(20, 10))
            plt.plot(range(1, len(myPCA.explained_variance_ratio_)+1),
                     myPCA.explained_variance_ratio_, alpha=0.8, marker='*', label="Explained Variance")
            y_label = plt.ylabel("Explained Variance")
            x_label = plt.xlabel("Components")
            plt.plot(range(1, len(myPCA.explained_variance_ratio_)+1), np.cumsum(
                myPCA.explained_variance_ratio_), c='r', marker='.', label="Cumulative Explained Variance")
            plt.legend()
            plt.title('Percentage of variance explained by component')
            st.pyplot(fig)

            st.header("Dataframe Standardized: Mean=0 SD=1")
            st.dataframe(df_standardized[numeric_cols].describe().T)

            num = st.slider(
                'Percentage of precision:', 0, 100, 90)
            total_explained_variance = myPCA.explained_variance_ratio_.cumsum()
            n_over = len(
                total_explained_variance[total_explained_variance >= num/100])
            n_to_reach_96 = total_explained_variance.shape[0] - n_over + 1
            st.text(
                f"To explain 96% of variance with PCS, we need the first {n_to_reach_96} principal components")

            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(myPCA.components_, cmap='seismic', xticklabels=list(df[numeric_cols].columns),
                        vmin=-np.max(np.abs(myPCA.components_)), vmax=np.max(np.abs(myPCA.components_)),
                        annot=True, ax=ax)
            ax.set_title("Weights that the PCA assigns to each component.")
            st.pyplot(fig)

                

                # posso selezionare solo le colonne di tipo numerico
                # numeric_cols = df.select_dtypes(include=['number'])

        st.header("Data Cleaning")
        columns = st.sidebar.multiselect("Enter the columns name to fill NaN with 0", df.columns)
        
        if st.button('Process data cleaning'):

            st.header('Fill NaN with 0')
            df1[columns] = df[columns].fillna(0)

            st.dataframe(df1)

            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                df1.to_excel(writer, index=False)
                # Close the Pandas Excel writer and output the Excel file to the buffer
                writer.save()
                st.download_button(
                    label="Download Excel Result",
                    data=buffer,
                    file_name="trasnformed_file.xlsx",
                    mime="application/vnd.ms-excel")
            

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import joblib
import numpy as np
import base64
from io import BytesIO
import os



def download_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.ms-excel;base64,{b64}" download="predicted_profit.xlsx">Download Excel</a>'
    return href

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def add_bg_image():
    st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('https://w.wallhaven.cc/full/g8/wallhaven-g89dgl.png');
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
    )

def main():

    #add_bg_image()
    model = joblib.load("3input_regression_startup.pkl")
    #inference
    st.title("Try the model") 

    st.subheader("Inference uploading a dataset")
    file = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])

    if file is not None:
        
        if os.path.splitext(file.name)[1] == ".xlsx":
            df = pd.read_excel(file, engine='openpyxl')
        else:
            df = pd.read_csv(file)
            if df['Profit'] is not None:
                df.drop(columns='Profit', inplace=True)


        df = df.round()
        st.dataframe(df)

        st.write('Dataframe Description')
        dfdesc = df.describe(include='all').T.fillna("")
        st.write(dfdesc)

        df_pred = model.predict(df)
        df_pred = df_pred.round()
        df['Profit'] = df_pred
        st.write('Updated Dataframe')
        st.dataframe(df)

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Predictions.csv',
            mime='text/csv',
        )

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

            download2 = st.download_button(
                label="Download data as Excel",
                data=buffer,
                file_name='Predictions.xlsx',
                mime='application/vnd.ms-excel'
            )
        


    if file is None:
        st.subheader("Inference with manual inputs")
        input1 = st.number_input("Enter a value for R&D Spend", value=0.00)
        input2 = st.number_input("Enter a float value Administration", value=0.00)
        input3 = st.number_input("Enter a float value Marketing Spend", value=0.00)
        final_input = np.array([input1, input2, input3])
        final_input = final_input.reshape(-1,3)
        pred = model.predict(final_input)
        pred_str = "%.2f" % pred[0]
        st.write("Prediction: ", pred_str)
        

if __name__ == '__main__':
    main()


# streamlit run app.py



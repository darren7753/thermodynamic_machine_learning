import os
import time
import tempfile
import numpy as np
import pandas as pd
import sweetviz as sv
import plotly.graph_objs as go
import streamlit as st
import streamlit.components.v1 as components

from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from pymongo import MongoClient
from pycaret.regression import *
from pygwalker.api.streamlit import StreamlitRenderer
from scipy.interpolate import griddata

# App Settings
st.set_page_config(
    page_title="ML Termodinamika",
    layout="wide"
)

st.markdown(
    """
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size:18px;
            }
        </style>
    """,
    unsafe_allow_html=True
)

# MongoDB Connections
load_dotenv(".env")
client = MongoClient(
    os.getenv("MONGO_CONNECTION_STRING"),
    serverSelectionTimeoutMS=300000
)
db = client[os.getenv("MONGO_DATABASE_NAME")]
collection_ENGINE = db[os.getenv("MONGO_COLLECTION_NAME_ENGINE")]
collection_SAVONIUS = db[os.getenv("MONGO_COLLECTION_NAME_SAVONIUS")]
collection_CRANK_MECHANISM = db[os.getenv("MONGO_COLLECTION_NAME_CRANK_MECHANISM")]
collection_PLTU = db[os.getenv("MONGO_COLLECTION_NAME_PLTU")]

# Functions
@st.cache_data
def load_data(collection_name):
    collection = db[collection_name]
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df.drop("_id", axis=1, inplace=True)
    return df

def store_to_mongo(df, collection_name):
    collection = db[collection_name]
    data_dict = [row for row in df.to_dict(orient="records")]
    collection.delete_many({})
    collection.insert_many(data_dict)
    st.cache_data.clear()

def append_to_mongo(df, collection_name):
    collection = db[collection_name]
    data_dict = [row for row in df.to_dict(orient="records")]
    collection.insert_many(data_dict)
    st.cache_data.clear()

def load_model_results(collection_name):
    model_dir = os.path.join("Models", collection_name)
    model_comparison_path = os.path.join(model_dir, "model_comparison.csv")
    best_model_path = os.path.join(model_dir, "best_model.txt")

    model_comparison = pd.read_csv(model_comparison_path, index_col=0)
    with open(best_model_path, "r") as f:
        best_model_name = f.read().strip()

    return model_comparison, best_model_name

# Session States
collection_map = {
    "Engine": os.getenv("MONGO_COLLECTION_NAME_ENGINE"),
    "Savonius": os.getenv("MONGO_COLLECTION_NAME_SAVONIUS"),
    "Crank Mechanism": os.getenv("MONGO_COLLECTION_NAME_CRANK_MECHANISM"),
    "PLTU": os.getenv("MONGO_COLLECTION_NAME_PLTU"),
}

target_map = {
    "Engine": "Effisiensi Panas (%)",
    "Savonius": "Rasio Kepesatan (l)",
    "Crank Mechanism": "Kerugian Gesekan %",
    "PLTU": "output generator (Watt)"
}

if "edit_mode" not in st.session_state:
    st.session_state["edit_mode"] = False
if "show_confirmation" not in st.session_state:
    st.session_state["show_confirmation"] = False
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Engine"
if "target" not in st.session_state:
    st.session_state["target"] = target_map[st.session_state["data_source"]]

def page_1():
    st.write("")
    st.write("")

    data_source = st.selectbox(
        label="Select Data Source",
        options=["Engine", "Savonius", "Crank Mechanism", "PLTU"],
        index=["Engine", "Savonius", "Crank Mechanism", "PLTU"].index(st.session_state["data_source"])
    )
    
    collection_name = collection_map[data_source]
    df = load_data(collection_name)
    st.session_state["df"] = df
    st.session_state["data_source"] = data_source
    st.session_state["target"] = target_map[data_source]

    tab1, tab2 = st.tabs(["Edit", "Upload"])
    with tab1:
        if st.session_state["edit_mode"]:
            edited_df = st.data_editor(st.session_state["df"], num_rows="dynamic", use_container_width=True)
            st.session_state["edited_df"] = edited_df
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col6:
                if st.button("Save", type="primary", use_container_width=True):
                    st.session_state["show_confirmation"] = True
                    st.rerun()
            with col1:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["edit_mode"] = False
                    st.rerun()
        else:
            st.dataframe(st.session_state["df"], use_container_width=True)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col6:
                if st.button("Edit", type="primary", use_container_width=True):
                    st.session_state["edit_mode"] = True
                    st.rerun()

        if st.session_state["show_confirmation"]:
            st.warning("Are you sure you want to save the changes?", icon="⚠️")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            if col6.button("Yes", type="primary", use_container_width=True):
                store_to_mongo(st.session_state["edited_df"], collection_name)
                st.toast("Data updated successfully!", icon="✅")
                time.sleep(2)
                st.session_state["edit_mode"] = False
                st.session_state["show_confirmation"] = False
                st.rerun()
            with col1:
                if st.button("No", use_container_width=True):
                    st.session_state["show_confirmation"] = False
                    st.rerun()

    with tab2:
        uploaded_file = st.file_uploader(
            label="Choose a CSV file",
            type=["csv"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            
            if list(df_uploaded.columns) == list(df.columns) and all(df_uploaded.dtypes == df.dtypes):
                st.dataframe(df_uploaded, use_container_width=True)
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                if col6.button("Upload", type="primary", use_container_width=True):
                    append_to_mongo(df_uploaded, collection_name)
                    st.toast("Data uploaded successfully!", icon="✅")
                    time.sleep(2)
                    st.session_state["uploaded_file"] = None
                    st.rerun()
            else:
                st.error("The columns or data types of the uploaded file do not match the current dataframe.",  icon="🚨")

def page_2():
    st.write("")
    st.write("")

    if "df" in st.session_state:
        tab1, tab2, tab3 = st.tabs(["Automated Visualization", "2D Visualization", "3D Visualization"])
        with tab1:
            analysis = sv.analyze(st.session_state["df"])
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                analysis.show_html(filepath=temp_file.name, open_browser=False, layout="vertical", scale=1.0)
                
                temp_file.seek(0)
                source_code = temp_file.read().decode("utf-8")

            components.html(source_code, height=1200, scrolling=True)

        with tab2:
            pyg_app = StreamlitRenderer(st.session_state["df"])
            pyg_app.explorer()

        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox(
                    label="X Axis",
                    options=st.session_state["df"].columns,
                    index=None
                )

            with col2:
                y_axis = st.selectbox(
                    label="Y Axis",
                    options=st.session_state["df"].columns,
                    index=None
                )

            with col3:
                z_axis = st.selectbox(
                    label="Z Axis",
                    options=st.session_state["df"].columns,
                    index=None
                )

            if (x_axis is not None) and (y_axis is not None) and (z_axis is not None):
                if (x_axis == y_axis) or (x_axis == z_axis) or (y_axis == z_axis):
                    st.error("X axis, Y axis, and Z axis cannot have the same values. Please select different columns.", icon="🚨")
                else:
                    col1, col2 = st.columns(2)

                    fig_contour = go.Figure(data=go.Contour(
                        z=st.session_state["df"][z_axis], 
                        x=st.session_state["df"][x_axis], 
                        y=st.session_state["df"][y_axis],
                        colorscale="Viridis"
                    ))
                    fig_contour.update_traces(contours_coloring="heatmap")
                    fig_contour.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_showgrid=False,
                        yaxis_showgrid=False
                    )

                    with col1:
                        st.subheader("Contour Plot")
                        with st.container(border=True):
                            st.plotly_chart(fig_contour, use_container_width=True)

                    x = np.array(st.session_state["df"][x_axis])
                    y = np.array(st.session_state["df"][y_axis])
                    z = np.array(st.session_state["df"][z_axis])

                    xi = np.linspace(x.min(), x.max(), 100)
                    yi = np.linspace(y.min(), y.max(), 100)

                    X, Y = np.meshgrid(xi, yi)
                    Z = griddata((x, y), z, (X, Y), method="cubic")

                    fig_surface = go.Figure(data=[go.Surface(
                        x=xi, 
                        y=yi, 
                        z=Z,
                        colorscale="Viridis"
                    )])
                    fig_surface.update_layout(
                        margin=dict(l=0, r=0, b=0, t=40)
                    )

                    with col2:
                        st.subheader("3D Surface Plot")
                        with st.container(border=True):
                            st.plotly_chart(fig_surface, use_container_width=True)

    else:
        st.error("No data loaded. Please load data from the 'Data' page.", icon="🚨")

def page_3():
    st.write("")
    st.write("")

    if "df" in st.session_state:
        df = st.session_state["df"].drop(st.session_state["target"], axis=1)
        collection_name = st.session_state["data_source"].replace(" ", "_").upper()

        try:
            model_comparison, best_model_name = load_model_results(collection_name)
        except FileNotFoundError:
            st.info("Currently, the models are unavailable for this dataset.", icon="ℹ️")
            return

        st.write("## Model Comparison")
        st.dataframe(model_comparison, use_container_width=True)
        st.write(f"Best Model: {best_model_name}")

        st.divider()

        st.write("## Prediction")
        model_names = model_comparison.index.tolist()
        selected_model = st.selectbox("Select Model", model_names)

        tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])
        with tab1:
            inputs = {}

            num_cols = len(df.columns)
            num_rows = (num_cols // 5) + (1 if num_cols % 5 != 0 else 0)

            for row in range(num_rows):
                cols = st.columns(5)
                for col in range(5):
                    col_idx = row * 5 + col
                    if col_idx < num_cols:
                        col_name = df.columns[col_idx]
                        if col_name != st.session_state["target"]:
                            inputs[col_name] = cols[col].number_input(
                                label=col_name, 
                                value=0.0, 
                                format="%.6f",
                                key=f"input_{col_name}"
                            )

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            if col6.button("Predict", key="predict_individual", type="primary", use_container_width=True):
                model_path = os.path.join("Models", collection_name, selected_model)
                model = load_model(model_path)
                prediction = model.predict(pd.DataFrame([inputs]))
                st.success(f"The predicted **{st.session_state['target']}** is **{prediction[0]}**", icon="✅")

        with tab2:
            uploaded_file = st.file_uploader(
                label="Choose a CSV file",
                type=["csv"],
                key="batch_file_uploader"
            )
            
            if uploaded_file is not None:
                df_batch = pd.read_csv(uploaded_file)
                df_predictions = None
                
                if list(df_batch.columns) == list(df.columns):
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    if col6.button("Predict", key="predict_batch", type="primary", use_container_width=True):
                        model_path = os.path.join("Models", collection_name, selected_model)
                        model = load_model(model_path)
                        predictions = model.predict(df_batch)
                        df_batch["Prediction"] = predictions
                        df_predictions = st.dataframe(df_batch, use_container_width=True)

                        if df_predictions:
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            col6.download_button(
                                label="Download Predictions",
                                data=df_batch.to_csv(index=False).encode("utf-8"),
                                file_name="predictions.csv",
                                mime="text/csv",
                                type="primary",
                                use_container_width=True
                            )
                else:
                    st.error("The columns of the uploaded file do not match the expected input columns.", icon="🚨")

    else:
        st.error("No data loaded. Please load data from the 'Data' page.", icon="🚨")

navbar = option_menu(
    menu_title=None,
    options=["Data", "EDA", "Prediksi"],
    icons=["database-fill", "bar-chart-fill", "robot"],
    default_index=0,
    orientation="horizontal"
)

if navbar == "Data":
    page_1()
elif navbar == "EDA":
    page_2()
elif navbar == "Prediksi":
    page_3()
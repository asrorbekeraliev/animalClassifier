import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform

import pathlib
pathlib.PosixPath = pathlib.WindowsPath
plt = platform.system()
if plt=='Linux':
    pathlib.WindowsPath = pathlib.PosixPath
    pathlib.PosixPath = pathlib.LinuxPath

#title
st.title('Animal classifier')
st.info('This model can classify only following three types of animals: Elephant, Zebra and Monkey')

# insert picture
file = st.file_uploader('Upload a picture', type=['png', 'jpg', 'jpeg', 'csv', 'svg'])

if file:
    img = PILImage.create(file)
    model = load_learner('animal_model.pkl')
    prediction, pd_id, probibility = model.predict(img)
    st.success(f"Prediction: {prediction}")
    st.info(f"Probibility: {probibility[pd_id]*100:.1f}")
    st.image(file)
    fig = px.bar(x=probibility*100, y=model.dls.vocab)
    st.plotly_chart(fig)

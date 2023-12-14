from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import pandas as pd

st.header("Klasifikasi Artikel Berita Dengan Reduksi Dimensi", divider='rainbow')
text = st.text_area("Masukkan Artikel Berita")

button = st.button("Submit")

if "svm_lda" not in st.session_state:
    # st.session_state.nb_reduksi = []
    # st.session_state.nb_asli = []
    st.session_state.svm_lda = []
    st.session_state.svm_asli = []

if button:
    vectorizer = joblib.load("resources/vectorizer.pkl")
    tfidf_matrics = vectorizer.transform([text]).toarray()
    
    # Predict Model Naive Bayes Reduksi
    # model_reduksi = joblib.load("resources/nbwithlda.pkl")
    # lda = joblib.load("resources/lda.pkl")
    # lda_transform = lda.transform(tfidf_matrics)
    # prediction_reduksi = model_reduksi.predict(lda_transform)
    # st.session_state.nb_reduksi = prediction_reduksi[0]
    
    # Predict SVM Reduksi
    model_reduksi = joblib.load("resources/svmwithlda.pkl")
    lda = joblib.load("resources/ldasvm.pkl")
    lda_transform = lda.transform(tfidf_matrics)
    prediction_reduksi = model_reduksi.predict(lda_transform)
    st.session_state.svm_lda = prediction_reduksi[0]
    
    # # Predict Model Naive Bayes Tanpa Reduksi
    # model_asli = joblib.load("resources/nbnonlda.pkl")
    # prediction_asli = model_asli.predict(tfidf_matrics)
    # st.session_state.nb_asli = prediction_asli[0]
    
    # Predict Model SVM Tanpa Reduksi
    model_asli = joblib.load("resources/svmnonlda.pkl")
    prediction_asli = model_asli.predict(tfidf_matrics)
    st.session_state.svm_asli = prediction_asli[0]

selected = option_menu(
  menu_title="",
  options=["Dataset Information" ,"Klasifikasi"],
  icons=["data", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Dataset Information":
    st.write("Dataset Asli")
    st.dataframe(pd.read_csv('resources/crawlingberitauas.csv'), use_container_width=True)
    # st.write("Dataset Hasil Reduksi Dimensi Naive Bayes")
    # st.dataframe(pd.read_csv('resources/hasilLDANB.csv'), use_container_width=True)
    st.write("Dataset Hasil Reduksi Dimensi SVM")
    st.dataframe(pd.read_csv('resources/hasilLDASVM.csv'), use_container_width=True)


elif selected == "Klasifikasi":
  if st.session_state.svm_lda:
      svm_lda, svm_NonLDA = st.tabs(["SVM(LDA)", "SVM(Tanpa LDA)"])
      
      # with nb_lda:
      #   st.write(f"Prediction Category : {st.session_state.nb_reduksi}")
        
      with svm_lda:
        st.write(f"Prediction Category : {st.session_state.svm_lda}")
        
      # with nb_NonLDA:
      #   st.write(f"Prediction Category : {st.session_state.nb_asli}")
      
      with svm_NonLDA:
        st.write(f"Prediction Category : {st.session_state.svm_asli}")

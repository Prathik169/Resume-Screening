import streamlit as st
import pickle 
import re
import nltk
import PyPDF2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# libraries for model fitting
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# librarie for metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report


nltk.download('punkt')
nltk.download('stopwords')

# Load models
try:
    with open('rf', 'rb') as rf_file:
        rf = pickle.load(rf_file)
except Exception as e:
    st.error(f"Error loading 'rf' model: {e}")
    st.stop()

try:
    with open('tfidf', 'rb') as tfidf_file:
        tfidfd = pickle.load(tfidf_file)
except Exception as e:
    st.error(f"Error loading 'tfidf' model: {e}")
    st.stop()
    
def resumeclean(x):
    CleanResume = re.sub('http\S+\s', '', x)
    CleanResume = re.sub('@\S+', '', CleanResume)
    CleanResume = re.sub('#\S+\s', '', CleanResume)
    CleanResume = re.sub('RT|cc', ' ', CleanResume)
    CleanResume = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', CleanResume)
    CleanResume = re.sub(r'[^\x00-\x7f]', ' ', CleanResume) 
    CleanResume = re.sub('\s+', ' ', CleanResume)
    
    return CleanResume

def main():
    st.title('Resume Screening App')
    st.markdown(
        "Welcome to the Resume Screening App! Upload your resume to predict the job category it best matches."
    )
    st.image("https://www.recruiterslineup.com/wp-content/uploads/2022/06/resume-screening-software.png", caption="Resume Screening", use_column_width=True)
        
    uploaded_file = st.file_uploader('Upload your resume', type=['txt', 'pdf', 'docx'])

    if uploaded_file:
        st.success("Resume uploaded successfully!")
        
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        else:
            text = uploaded_file.getvalue().decode("utf-8")
        
        cleaned_resume = resumeclean(text)
        cleaned_resume = tfidfd.transform([cleaned_resume])
        prediction_id = rf.predict(cleaned_resume)[0]

        category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.subheader("Prediction Result:")
        st.success(f"The predicted category is: {category_name}")

    else:
        st.warning("Please upload a valid file.")

# Python main
if __name__ == '__main__':
    main()

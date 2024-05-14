import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pycaret.classification import *
from pycaret.regression import *

def read_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)
    elif file.name.endswith('.json'):
        df = pd.read_json(file)
    else:
        raise ValueError("الملف غير مدعوم. CSV، Excel، أو JSON.")
    return df
def drop_columns(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop)

    return df
def perform_analysis(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            st.subheader(f"تحليل العمود الرقمي: {col}")
            st.write("المتوسط:", df[col].mean())
            st.write("القيمة القصوى:", df[col].max())
            st.write("القيمة الدنيا:", df[col].min())
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.histplot(df[col]))
            st.pyplot()
        else:
            # تنفيذ الوظائف النصية
            st.subheader(f"تحليل العمود النصي: {col}")
            st.write("القيم المميزة:", df[col].unique())
            st.write("عدد القيم المميزة:", df[col].nunique())


#perform_analysis(df)


uploaded_file = st.file_uploader("تحميل ملف البيانات", type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file is not None:
    df = read_data(uploaded_file)
    st.write(df)
columns_to_drop = st.multiselect("اختر الأعمدة التي تريد حذفها", df.columns)
    
if columns_to_drop:
        df = drop_columns(df, columns_to_drop)
        st.write("البيانات بعد حذف الأعمدة المحددة:")
        st.write(df)
else:
        st.write("لم يتم تحديد أي أعمدة للحذف.")

perform_eda_checkbox = st.checkbox("هل ترغب في أداء تحليل البيانات التكريمي (EDA)?")
if perform_eda_checkbox:
    columns_to_analyze = st.multiselect("اختر الأعمدة التي ترغب في تحليلها", df.columns)
    if columns_to_analyze:
            perform_analysis(df[columns_to_analyze])
    else:
            st.write("لم يتم تحديد أي أعمدة لتحليلها.")





##
handle_missing_values_option = st.radio("كيف ترغب في التعامل مع القيم المفقودة؟", ["حذف الصفوف ذات القيم المفقودة", "ملء القيم المفقودة بقيمة محددة", "إضافة الفئة الإضافية"])

if handle_missing_values_option == "حذف الصفوف ذات القيم المفقودة":
    # قم بحذف الصفوف ذات القيم المفقودة
    df.dropna(inplace=True)
    st.write("تم حذف الصفوف ذات القيم المفقودة.")
    st.write(df)
elif handle_missing_values_option == "ملء القيم المفقودة بقيمة محددة":
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            filling_method = st.radio(f"اختر الطريقة لملء القيم المفقودة في العمود {col}", ["المتوسط (Mean)", "الوسيط (Median)", "الوضع (Mode)"])
            if filling_method == "المتوسط (Mean)":
                filling_value = df[col].mean()
            elif filling_method == "الوسيط (Median)":
                filling_value = df[col].median()
            else:  # الوضع (Mode)
                filling_value = df[col].mode()[0]
            df[col].fillna(filling_value, inplace=True)
            st.write(f"تم ملء القيم المفقودة في العمود {col} بقيمة: {filling_value}")
        else:
            filling_value = df[col].mode()[0] 
            df[col].fillna(filling_value, inplace=True)
            st.write(f"تم ملء القيم المفقودة في العمود {col} بقيمة: {filling_value}")
    st.write(df)
elif handle_missing_values_option == "إضافة الفئة الإضافية":
    # إضافة فئة إضافية 
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna("Missing", inplace=True)
    st.write("تمت إضافة فئة إضافية للقيم المفقودة في الأعمدة الفئوية.")
    st.write(df)



######
is_classification = False
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if len(df[col].unique()) == 2 and df[col].min() == 0 and df[col].max() == 1:
            is_classification = True
            break

if is_classification:
    st.write("المهمة  (Classification)")
    # اختيار X و Y
    X_options = df.drop(columns=[col]).columns
    Y_options = df[col].name
else:
    st.write("المهمة هي (Regression)")
    # اختيار X و Y
    X_options = df.drop(columns=[col]).columns
    Y_options = df[col].name

X = st.multiselect("اختر المتغيرات المستقلة (X)", options=X_options)
numeric_columns = df.select_dtypes(include=['int', 'float']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
Y_options = list(numeric_columns) + list(categorical_columns)
Y = st.multiselect("اختر المتغير المستجيب (Y)", options=Y_options)


if is_classification:
    classifiers = {
        "Support Vector Classifier": SVC(),
        "K Neighbors Classifier": KNeighborsClassifier(),
        "Random Forest Classifier": RandomForestClassifier()
    }
else:
    regressors = {
        "Linear Regression": LinearRegression()
    }

if is_classification:
    X_train, X_test, y_train, y_test = train_test_split(df[X], df[Y], test_size=0.2, random_state=0)
else:
    X_train, X_test, y_train, y_test = train_test_split(df[X], df[Y], test_size=0.2, random_state=0)

if is_classification:
    st.write("التحليل الإحصائي والرسم البياني (Classification):")
    for classifier_name, classifier in classifiers.items():
        st.write(f"نتائج {classifier_name}:")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        st.write("تقرير التصنيف:")
        st.write(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
else:
    # Regression analysis
    st.write("التحليل الإحصائي والرسم البياني (Regression):")
    for regressor_name, regressor in regressors.items():
        st.write(f"نتائج {regressor_name}:")
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        st.write("مقدار Mean Squared Error:")
        st.write(mean_squared_error(y_test, y_pred))
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, y_pred, color='red')
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.title("Regression Plot")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot()


    ####

st.write("اختيار النموذج الأفضل ولكن تأكد أن البيانات جميعها متناسبة مع بعضها")
setup(data=df, target=Y, session_id=123)
best_model = compare_models()

if best_model:
    st.write("Best model retrieved:", best_model)
else:
    st.write("No model available for display")


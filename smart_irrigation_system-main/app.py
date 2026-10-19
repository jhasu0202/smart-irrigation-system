import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================================

st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="wide"
)

# =========================================
# STYLE
# =========================================

st.markdown("""
<style>

.main{
background-color:#f5f7fa;
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#2E7D32;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
}

div[data-testid="metric-container"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 3px 10px rgba(0,0,0,.1);
}

.stButton>button{
width:100%;
height:55px;
font-size:22px;
font-weight:bold;
border-radius:12px;
}

</style>
""",unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.markdown(
"<div class='title'>🌱 AI SMART IRRIGATION SYSTEM</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>Machine Learning Based Water Prediction Dashboard</div>",
unsafe_allow_html=True
)

st.write("---")

# =========================================
# LOAD DATA
# =========================================

df=pd.read_csv(
"smart_irrigation_system.csv"
)

encoders={}

cols=[
'Soil_Type',
'Crop_Type',
'Crop_Growth_Stage',
'Season',
'Irrigation_Type',
'Water_Source',
'Region',
'Mulching_Used',
'Irrigation_Need'
]

for col in cols:

    le=LabelEncoder()

    df[col]=le.fit_transform(
    df[col]
    )

    encoders[col]=le

# =========================================
# MODEL
# =========================================

X=df.drop(
"Irrigation_Need",
axis=1
)

y=df[
"Irrigation_Need"
]

X_train,X_test,y_train,y_test=\
train_test_split(
X,
y,
test_size=.2,
random_state=42
)

model=RandomForestClassifier()

model.fit(
X_train,
y_train
)

pred=model.predict(
X_test
)

accuracy=accuracy_score(
y_test,
pred
)

# =========================================
# DASHBOARD CARDS
# =========================================

a,b,c=st.columns(3)

a.metric(
"Dataset Records",
len(df)
)

b.metric(
"Features",
X.shape[1]
)

c.metric(
"Accuracy",
f"{accuracy*100:.2f}%"
)

st.write("---")

# =========================================
# INPUTS
# =========================================

left,right=st.columns(2)

with left:

    st.subheader(
    "🌾 Soil Information"
    )

    moisture=st.slider(
    "Soil Moisture %",
    0,100,35
    )

    ph=st.slider(
    "Soil pH",
    0.0,14.0,6.5
    )

    temp=st.slider(
    "Temperature °C",
    0,50,30
    )

with right:

    st.subheader(
    "🌦 Weather Information"
    )

    humidity=st.slider(
    "Humidity %",
    0,100,65
    )

    rainfall=st.slider(
    "Rainfall mm",
    0,300,10
    )

    crop=st.selectbox(
    "Crop Type",
    ["Rice","Wheat","Cotton","Maize"]
    )

st.write("---")

# =========================================
# PREDICTION
# =========================================

if st.button(
"🚀 Predict Irrigation Need"
):

    crop_map={

    "Rice":0,
    "Wheat":1,
    "Cotton":2,
    "Maize":3

    }

    sample=pd.DataFrame({

'Soil_Type':[1],
'Soil_pH':[ph],
'Soil_Moisture':[moisture],
'Organic_Carbon':[1.2],
'Electrical_Conductivity':[0.5],
'Temperature_C':[temp],
'Humidity':[humidity],
'Rainfall_mm':[rainfall],
'Sunlight_Hours':[8],
'Wind_Speed_kmh':[12],
'Crop_Type':[crop_map[crop]],
'Crop_Growth_Stage':[1],
'Season':[0],
'Irrigation_Type':[1],
'Water_Source':[0],
'Field_Area_hectare':[2.5],
'Mulching_Used':[1],
'Previous_Irrigation_mm':[20],
'Region':[2]

})

    result=model.predict(
    sample
    )[0]

    result=encoders[
    "Irrigation_Need"
    ].inverse_transform(
    [result]
    )[0]

    st.subheader(
    "Prediction Result"
    )

    if result=="High":

        st.error(
        "🔴 HIGH IRRIGATION REQUIRED"
        )

        st.progress(90)

    elif result=="Medium":

        st.warning(
        "🟡 MEDIUM IRRIGATION REQUIRED"
        )

        st.progress(60)

    else:

        st.success(
        "🟢 LOW IRRIGATION REQUIRED"
        )

        st.progress(30)

    st.info(
    f"Recommended action based on current environmental conditions for {crop}"
    )

st.write("---")

st.subheader(
"📊 Irrigation Distribution"
)

fig,ax=plt.subplots()

df["Irrigation_Need"].value_counts().plot(
kind='bar',
ax=ax
)

st.pyplot(
fig
)

st.success(
"System Running Successfully ✅"
)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.title("California Housing Price Predictor")
st.markdown("### Предсказание медианной стоимости дома на основе характеристик района")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    # Заполняем пропуски
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    # Кодируем категориальный признак
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean')
    return df

df = load_data()

# Боковая панель с параметрами модели
st.sidebar.header("Настройки модели")
n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 
                                   min_value=10, max_value=300, value=100, step=10)
max_depth = st.sidebar.slider("Максимальная глубина (max_depth)",
                                min_value=5, max_value=50, value=20, step=5)

# Подготовка данных
target = 'median_house_value'
features = [col for col in df.columns if col != target]

X = df[features].copy()
y = df[target].copy()

# Масштабирование и обучение
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=n_estimators, 
                              max_depth=max_depth,
                              random_state=42,
                              n_jobs=-1)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.sidebar.markdown("---")
st.sidebar.metric("MAE (на тесте)", f"${mae:,.0f}")
st.sidebar.metric("R² (на тесте)", f"{r2:.3f}")

# Интерфейс для предсказания
st.subheader("Предсказание стоимости для нового района")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Основные параметры**")
    longitude = st.number_input("Долгота (longitude)", 
                                 min_value=-124.5, max_value=-114.0, value=-122.0, step=0.1)
    latitude = st.number_input("Широта (latitude)", 
                                min_value=32.5, max_value=42.0, value=37.5, step=0.1)
    housing_median_age = st.number_input("Медианный возраст домов (лет)", 
                                          min_value=1, max_value=52, value=28, step=1)
    total_rooms = st.number_input("Всего комнат", 
                                   min_value=1, max_value=40000, value=2000, step=100)
    
with col2:
    st.markdown("**Жилищные условия**")
    total_bedrooms = st.number_input("Всего спален", 
                                      min_value=1, max_value=7000, value=400, step=50)
    population = st.number_input("Население района", 
                                  min_value=1, max_value=40000, value=1000, step=100)
    households = st.number_input("Домохозяйств", 
                                  min_value=1, max_value=7000, value=400, step=50)
    median_income = st.number_input("Медианный доход (x10,000)", 
                                     min_value=0.5, max_value=15.0, value=4.0, step=0.1)

# Категориальный признак отдельно
st.subheader("Расположение относительно океана")
ocean_options = {
    '<1H OCEAN': 'Менее 1 часа до океана',
    'INLAND': 'Внутри континента',
    'NEAR OCEAN': 'Близко к океану',
    'NEAR BAY': 'Близко к заливу',
    'ISLAND': 'На острове'
}
ocean_proximity = st.selectbox("Тип расположения", 
                                options=list(ocean_options.keys()),
                                format_func=lambda x: ocean_options[x])

# Кнопка предсказания
if st.button("Рассчитать стоимость", type="primary"):
    # Создаём входные данные
    input_data = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income
    ]], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                 'total_bedrooms', 'population', 'households', 'median_income'])
    
    # Добавляем one-hot признаки
    for ocean_type in ocean_options.keys():
        input_data[f'ocean_{ocean_type}'] = 1 if ocean_proximity == ocean_type else 0
    
    # Убеждаемся, что колонки совпадают с обучающими
    for col in features:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[features]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"Прогнозируемая стоимость дома: **${prediction:,.0f}**")
    
    # Интервал уверенности
    st.info(f"Типичный диапазон для похожих районов: **${prediction-50000:,.0f} - ${prediction+50000:,.0f}**")
    
    # Интерпретация
    if prediction < 150000:
        st.caption("Низкая стоимость: доступное жильё")
    elif prediction < 300000:
        st.caption("Средняя стоимость: стандартное жильё")
    elif prediction < 500000:
        st.caption("Выше среднего: комфортное жильё")
    else:
        st.caption("Премиум-класс: элитное жильё")

# График важности признаков
st.subheader("Важность признаков в модели")
feature_importance = pd.DataFrame({
    'Признак': features,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=True)

st.bar_chart(feature_importance.set_index('Признак'))

# Таблица с данными
with st.expander("Показать примеры данных"):
    st.dataframe(df.head(10))
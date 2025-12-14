import requests
import streamlit as st
import pandas as pd
import json
import time

API_BASE_URL = "http://localhost:8000"

def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            return True
        return False
    except:
        return False

# настройка страницы
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide"
)

#вкладки
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Главная", "🎓 Обучение", "🔮 Предсказания", "⚙️ Управление"])

with tab1:
    # заголовок главной страницы
    st.title("ML Models Dashboard")
    
    # статус подключения
    st.header("🔗 Статус подключения")
    if check_api_status():
        st.success("✅ API сервис доступен!")
    else:
        st.error("❌ API сервис недоступен")
        st.info("Убедитесь, что API запущен на localhost:8000")
    
    st.markdown("---")
    
    # Суммари возможностей дашборда
    st.header("Обзор возможностей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Обучение моделей")
        st.markdown("""
        - **Выбор алгоритма**: Logistic Regression, Random Forest
        - **Настройка гиперпараметров**: регуляризация и её сила, количество иттераций, количество и глубина деревьев и т.д.
        - **Загрузка данных**
        - **Автоматическое сохранение**: ID модели для дальнейшего использования
        """)
    with col2:    
        st.subheader("🔮 Предсказания")
        st.markdown("""
        - **Выбор обученной модели**: из списка доступных
        - **Визуализация**: таблицы и графики результатов
        - **Экспорт данных**: экспорт csv файла 
        """)
    
    st.subheader("⚙️ Управление моделями")
    st.markdown("""
    - **Просмотр всех моделей**: полный список обученных ML-моделей
    - **Мониторинг статуса**: отслеживание состояния моделей
    """)
    
    st.markdown("---")
    st.header("Начало работы")
    
    st.info("""
    **Чтобы начать работу:**
    1. Перейдите во вкладку **"Обучение"**
    2. Выберите тип модели и настройте параметры
    3. Введите данные для обучения в нужном формате
    4. Нажмите "Обучить модель" 
    5. Используйте ID модели для предсказаний во вкладке **"Предсказания"**
    """)

with tab2:
    st.header("🎓 Обучение моделей")
    
    # доступные модели
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        available_models = response.json()["models"] if response.status_code == 200 else ["logreg", "rf"]
    except:
        available_models = ["logreg", "rf"]
    
    model_type = st.selectbox("Выберите модель для обучения:", available_models)
    
    # гиперпараметры
    st.subheader("Гиперпараметры")
    

    if model_type == "logreg":
        
        penalty = st.selectbox("Тип регуляризации", ["l2", "l1"], 
                            help="l1 - Lasso (отбор признаков), l2 - Ridge (сжатие весов)")

        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Параметр регуляризации C", 0.1, 10.0, 1.0, 0.1, 
                        help="Сила регуляризации: меньше = сильнее регуляризация")
        
        with col2:
            max_iter = st.slider("Максимум итераций", 100, 1000, 100,
                            help="Лимит итераций для схождения алгоритма")
        
        hyperparams = {
            "C": C,
            "penalty": penalty, 
            "max_iter": max_iter,
        }

    else:  # random forest
        
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Количество деревьев", 10, 200, 100)
            max_depth = st.slider("Макс. глубина дерева", 2, 20, 10,
                                help="Ограничение глубины для предотвращения переобучения")
        
        with col2:
            min_samples_split = st.slider("Мин. samples для разделения", 2, 10, 2,
                                        help="Минимальное количество samples для разделения узла")
            random_state = st.number_input("Random state", 0, 100, 42,
                                         help="Для воспроизводимости результатов")
        
        hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state
        }

    # данные для обучения
    st.subheader("Данные для обучения")
    
    st.info("💡 Пример данных для обучения:")
    st.code("""
# Features - список списков чисел
# Target - список чисел (0 или 1 для классификации)
{
  "features": [[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, 3.0]],
  "target": [0, 1, 0, 1]
}
    """)
    
    features_input = st.text_area(
        "Features (список списков чисел):",
        value='[[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, 3.0]]',
        height=100
    )
    
    target_input = st.text_area(
        "Target (список чисел):",
        value='[0, 1, 0, 1]',
        height=80
    )
    
    if st.button("Обучить модель", key="train_button"):
        try:
            features = json.loads(features_input)
            target = json.loads(target_input)
            
            payload = {
                "model_name": model_type,
                "hyperparams": hyperparams,
                "features": features,
                "target": target
            }
            
            with st.spinner("Обучаем модель... Это может занять несколько секунд"):
                response = requests.post(f"{API_BASE_URL}/train", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    model_id = result["model_id"]
                    
                    st.success("✅ Модель успешно обучена!")
                    st.info(f"**ID модели:** `{model_id}`")
                    
                    # сохраняем ID модели в session state
                    if "trained_models" not in st.session_state:
                        st.session_state.trained_models = []
                    st.session_state.trained_models.append({
                        "id": model_id,
                        "type": model_type,
                        "timestamp": time.time()
                    })
                    
                else:
                    st.error(f"❌ Ошибка при обучении: {response.text}")
                    
        except Exception as e:
            st.error(f"❌ Ошибка в формате данных: {e}")

with tab3:
    st.header("🔮 Предсказания")
    
    # проверяем есть ли обученные модели
    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.warning("📝 Сначала обучите модель в разделе 'Обучение'")
    else:
        # выбор модели для предсказания
        model_options = [f"{model['id']} ({model['type']})" for model in st.session_state.trained_models]
        selected_model_str = st.selectbox("Выберите модель для предсказания:", model_options)
        
        # извлекаем ID модели из выбранной строки
        selected_model_id = selected_model_str.split(" ")[0]
        
        st.subheader("Данные для предсказания")
        
        st.info("💡 Введите данные в том же формате, что и при обучении:")
        st.code("""
# Features - список списков чисел
{
  "features": [[1.5, 2.5], [3.5, 4.5]]
}
        """)
        
        prediction_features = st.text_area(
            "Features для предсказания:",
            value='[[1.5, 2.5], [3.5, 4.5]]',
            height=100,
            key="pred_features"
        )
        
        if st.button("Сделать предсказание", key="predict_button"):
            try:
                features = json.loads(prediction_features)
                
                payload = {
                    "model_id": selected_model_id,
                    "features": features
                }
                
                with st.spinner("Выполняем предсказание..."):
                    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predictions = result["predictions"]
                        
                        st.success("✅ Предсказания получены!")
                        
                        # формируем таблицу
                        results_data = []
                        for i, (feature, prediction) in enumerate(zip(features, predictions)):
                            results_data.append({
                                "№": i + 1,
                                "Входные данные": str(feature),
                                "Предсказание": prediction
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # визуализация
                        if len(predictions) > 1:
                            st.subheader("Визуализация предсказаний")
                            prediction_counts = pd.Series(predictions).value_counts().sort_index()
                            st.bar_chart(prediction_counts)
                        
                    else:
                        st.error(f"❌ Ошибка при предсказании: {response.text}")
                        
            except Exception as e:
                st.error(f"❌ Ошибка в формате данных: {e}")

with tab4:
    st.header("⚙️ Управление моделями")
    
    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.info("Нет обученных моделей. Сначала обучите модель в разделе 'Обучение'.")
    else:
        st.write(f"**Обучено моделей:** {len(st.session_state.trained_models)}")
        
        for i, model in enumerate(st.session_state.trained_models):
            with st.expander(f"Модель: {model['id']} ({model['type']})", expanded=True):
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**ID:** `{model['id']}`")
                    st.write(f"**Тип:** {model['type']}")
                    
                with col2:
                    st.write("**Статус:** ✅ Обучена")
                
                with col3:
                    if st.button("🗑️ Удалить", key=f"delete_{i}"):
                        try:
                            response = requests.delete(f"{API_BASE_URL}/models/{model['id']}")
                            if response.status_code == 200:
                                # Удаляем модель из session state
                                st.session_state.trained_models.pop(i)
                                st.success(f"✅ Модель {model['id']} удалена")
                                st.rerun()
                            else:
                                st.error(f"❌ Ошибка при удалении: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Ошибка: {e}")


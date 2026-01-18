# HW06 – Report

## 1. Dataset

- Использовал `S06-hw-dataset-01.csv` (бинарная классификация).
- Размер: 12 000 строк × 30 столбцов (включая `id` и целевой столбец).
- Целевая переменная `target`: класс `0` — 67.7% (8119 наблюдений), класс `1` — 32.3% (3881 наблюдение).
- Признаки в основном числовые (`num01`–`num24`, `tenure_months`), есть несколько категориальных-подобных признаков с небольшим числом уникальных значений (`cat_contract`, `cat_region`, `cat_payment`), закодированных целыми.

## 2. Protocol

- Разбиение: `train_test_split` с `test_size=0.2`, `random_state=42`, `stratify=y`, чтобы сохранить исходный дисбаланс на тесте.
- Подбор гиперпараметров: `GridSearchCV` на train с `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` и оптимизацией по ROC-AUC.
- Метрики: accuracy, F1 и ROC-AUC. Accuracy полезна для общего понимания, F1 отражает баланс precision/recall при умеренном дисбалансе, ROC-AUC — основная метрика сравнения, позволяющая учесть качество ранжирования.

## 3. Models

- `DummyClassifier(strategy='most_frequent')` — базовый ориентир.
- `LogisticRegression` внутри `Pipeline` со `StandardScaler`, `class_weight='balanced'`, без подбора гиперпараметров (baseline с регуляризацией по умолчанию).
- `DecisionTreeClassifier(class_weight='balanced')` с подбором `max_depth ∈ {4, 6, None}`, `min_samples_leaf ∈ {1, 10}`, `ccp_alpha ∈ {0, 0.0005}`.
- `RandomForestClassifier` с 300 деревьями, `class_weight='balanced_subsample'`; подбирались `max_depth ∈ {None, 16}`, `min_samples_leaf ∈ {1, 5}`, `max_features ∈ {'sqrt', 0.4}`.
- `HistGradientBoostingClassifier(class_weight='balanced')` с подбором `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, None}`, `max_leaf_nodes=31`, `min_samples_leaf ∈ {20, 50}`, `l2_regularization=0`.

## 4. Results

| Model                 | Accuracy | F1     | ROC-AUC |
|-----------------------|---------:|-------:|--------:|
| DummyClassifier       | 0.677    | 0.000  | 0.500   |
| LogisticRegression    | 0.805    | 0.722  | 0.876   |
| DecisionTree          | 0.863    | 0.792  | 0.903   |
| RandomForest          | 0.925    | 0.879  | 0.968   |
| HistGradientBoosting  | **0.935**| **0.900**| **0.973** |

Победитель — `HistGradientBoostingClassifier`: он показал лучший ROC-AUC (0.973) и одновременно улучшил accuracy/F1 относительно остальных ансамблей, что подтверждает его устойчивость на дисбалансных данных.

## 5. Analysis

- **Устойчивость.** Запустил 5 прогонов с `random_state ∈ {0,…,4}` для `LogisticRegression` и настроенного HGB. Логистическая регрессия давала ROC-AUC в диапазоне 0.876–0.890 (mean 0.883), а HGB — 0.969–0.973 (mean 0.971) при accuracy≈0.935±0.003. Разброс HGB заметно ниже, что подтверждает стабильность ансамбля.
- **Ошибки.** Confusion matrix лучшей модели (`TN=1545`, `FP=79`, `FN=76`, `TP=700`) показывает сопоставимые количества ложноположительных/ложноотрицательных примеров; основная часть ошибок — пограничные объекты, что видно по плавным ROC/PR кривым.
- **Интерпретация.** Permutation importance (top-10) выделил `num18`, `num19`, `num07`, `num04`, `num24` и `num20` как основные факторы. Они дают наибольший спад ROC-AUC при перетасовке, что согласуется с наблюдениями по распределениям признаков: именно эти числовые фичи имели наибольший разрыв между классами.

## 6. Conclusion

- Даже при умеренном дисбалансе accuracy вводит в заблуждение: ROC-AUC и F1 лучше показывают реальное качество.
- Контроль сложности деревьев (через `min_samples_leaf`, `max_depth`, `ccp_alpha`) заметно повышает обобщающую способность по сравнению с “сырым” деревом.
- Ансамбли (RF, HGB) выигрывают у одиночных моделей благодаря снижению вариативности и учёту сложных взаимодействий.
- Честный протокол с фиксированным train/test и CV на train позволяет прозрачно сравнивать модели и сохранять воспроизводимые артефакты.

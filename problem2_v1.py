import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

print('''

############################################################
#  ИНФОРМАЦИЯ ПО ПОЛИНОМИАЛЬНОЙ РЕГРЕССИИ С РЕГУЛЯРИЗАЦИЕЙ #
############################################################
''')
print()
print("―――――――――――――――――――――――――――――――――――")
print("1. СРАВНЕНИЕ МЕТОДОВ РЕГУЛЯРИЗАЦИИ:")
print("―――――――――――――――――――――――――――――――――――")
print("L1 (Lasso): Создает разреженные модели, автоматический отбор признаков")
print("L2 (Ridge): Уменьшает переобучение, не создает разреженность")
print("ElasticNet: Компромисс между L1 и L2, гибкость через l1_ratio")
print("Без регуляризации: Сильное переобучение на полиномиальных признаках")
print()
print("――――――――――――――――――――")
print("2. ВЫБОР ПАРАМЕТРОВ:")
print("――――――――――――――――――――")
print("Alpha (сила регуляризации):")
print("  - Слишком маленький: слабый эффект регуляризации")
print("  - Слишком большой: недообучение, все веса стремятся к нулю")
print("  - Рекомендация: начинать с 0.01, использовать кросс-валидацию")
print()
print("l1_ratio (для ElasticNet):")
print("  - 0.0: эквивалентно L2 регуляризации")
print("  - 1.0: эквивалентно L1 регуляризации")
print("  - 0.5: равный вклад L1 и L2 (по умолчанию)")
print("  - Рекомендация: 0.5 как стартовое значение")
print()
print("―――――――――――――――――――――――――――――")
print("3. СУЩЕСТВУЮЩИЕ РЕКОМЕНДАЦИИ:")
print("―――――――――――――――――――――――――――――")
print("1. Всегда использовать регуляризацию для полиномиальных моделей")
print("2. Начинать с ElasticNet (alpha=0.01, l1_ratio=0.5)")
print("3. Использовать стандартизацию признаков перед регуляризацией")
print("4. Для интерпретируемости моделей использовать L1 регуляризацию")
print("5. Для максимизации качества предсказаний использовать ElasticNet или L2")
print("6. Использовать кросс-валидацию для подбора alpha и l1_ratio")
print("7. Визуализировать распределение весов для диагностики")
print()
print("―――――――――――――――――――――――――――――")
print("4. ПРЕИМУЩЕСТВА И НЕДОСТАТКИ:")
print("―――――――――――――――――――――――――――――")
print()
print("L1 (Lasso):")
print("  + Создает разреженные модели (отбор признаков)")
print("  + Хорошая интерпретируемость")
print("  - Может исключить коррелированные признаки произвольно")
print("  - Менее стабилен при небольших изменениях данных")
print()
print("L2 (Ridge):")
print("  + Стабильность и надежность")
print("  + Эффективно борется с мультиколлинеарностью")
print("  + Всегда дает решение")
print("  - Не создает разреженность")
print("  - Все признаки остаются в модели")
print()
print("ElasticNet:")
print("  + Гибкость через l1_ratio")
print("  + Сочетает преимущества L1 и L2")
print("  + Хорошо работает с коррелированными признаками")
print("  - Дополнительный гиперпараметр для настройки")
print()

print('''

###############
#  ОБРАБОТКА  #
###############
''')
print()
print("1. Загрузка и подготовка данных Wine Quality...")
data_csv_url = \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
datFra = pd.read_csv(data_csv_url, sep=';')

# Разделение на признаки и целевую переменную
X = datFra.drop(labels='quality', axis=1).values
y = datFra['quality'].values.reshape(-1, 1)

print(f"Исходная размерность признаков: {X.shape[1]}")
print(f"Количество образцов: {X.shape[0]}")
print()
print("2. Создание полиномиальных признаков (степень 2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Размерность после полиномиального преобразования: {X_poly.shape[1]}")

# Нормализация данных
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_poly_scaled = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_scaled, y_scaled, test_size=0.2, random_state=42
)

# Добавление свободного члена (intercept)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
print(f"Общая размерность признаков (включая intercept): {X_train.shape[1]}")


class PolynomialRegressionSGD:
    def __init__(self, n_features, learning_rate=0.01, batch_size=32,
                 regularization='l2', alpha_=0.01, l1_ratio_=0.5):
        """
        Полиномиальная регрессия с SGD и регуляризацией

        Параметры:
        - regularization: 'l1', 'l2', 'elasticnet'
        - alpha: коэффициент регуляризации
        - l1_ratio: соотношение L1/L2 для elasticnet (0-1)
        """
        self.theta = np.random.randn(n_features, 1) * 0.01
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha_
        self.l1_ratio = l1_ratio_
        self.loss_history = []
        self.train_r2_history = []

    def compute_loss(self, X_, y_, include_reg=True):
        """Вычисление функции потерь с регуляризацией"""
        m = len(y_)
        predictions = X_.dot(self.theta)

        loss_mse = (1 / (2 * m)) * np.sum((predictions - y_) ** 2)

        # Регуляризация (не применяется к intercept)
        loss_reg = 0
        if include_reg and self.alpha > 0:
            theta_no_intercept = self.theta[1:]  # исключаем intercept

            if self.regularization == 'l1':
                loss_reg = self.alpha * np.sum(np.abs(theta_no_intercept))
            elif self.regularization == 'l2':
                loss_reg = self.alpha * np.sum(theta_no_intercept ** 2)
            elif self.regularization == 'elasticnet':
                loss_l1 = self.l1_ratio * np.sum(np.abs(theta_no_intercept))
                loss_l2 = (1 - self.l1_ratio) * np.sum(theta_no_intercept ** 2)
                loss_reg = self.alpha * (loss_l1 + loss_l2)

        return loss_mse + loss_reg

    def compute_gradient(self, X_batch, y_batch):
        """Вычисление градиента с учетом регуляризации"""
        m_batch = len(y_batch)
        predictions = X_batch.dot(self.theta)
        error = predictions - y_batch

        gradient_mse = X_batch.T.dot(error) / m_batch

        # Добавление регуляризации (не применяется к intercept)
        if self.alpha > 0:
            gradient_reg = np.zeros_like(self.theta)
            theta_no_intercept = self.theta[1:].copy()

            if self.regularization == 'l1':
                # Субградиент для L1 (используем знаковую функцию)
                gradient_reg[1:] = self.alpha * np.sign(theta_no_intercept)
            elif self.regularization == 'l2':
                gradient_reg[1:] = 2 * self.alpha * theta_no_intercept
            elif self.regularization == 'elasticnet':
                gradient_l1 = self.l1_ratio * np.sign(theta_no_intercept)
                gradient_l2 = 2 * (1 - self.l1_ratio) * theta_no_intercept
                gradient_reg[1:] = self.alpha * (gradient_l1 + gradient_l2)

            gradient_mse += gradient_reg

        return gradient_mse

    def fit(self, X_, y_, n_epochs=200, learning_rate_schedule=None,
            early_stopping_patience=10, verbose=False):
        """Обучение модели"""
        m = len(y_)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            if learning_rate_schedule:
                lr = learning_rate_schedule(epoch, self.learning_rate)
            else:
                lr = self.learning_rate

            indices = np.random.permutation(m)
            # noinspection PyPep8Naming
            X_shuffled = X_[indices]
            y_shuffled = y_[indices]

            for j in range(0, m, self.batch_size):
                # noinspection PyPep8Naming
                X_batch = X_shuffled[j:j + self.batch_size]
                y_batch = y_shuffled[j:j + self.batch_size]

                gradient = self.compute_gradient(X_batch, y_batch)
                self.theta -= lr * gradient

            train_loss = self.compute_loss(X_, y_)
            self.loss_history.append(train_loss)

            train_predicted = self.predict(X_)
            train_r2 = r2_score(y_, train_predicted)
            self.train_r2_history.append(train_r2)

            if train_loss < best_loss:  # Ранняя остановка
                best_loss = train_loss
                patience_counter = 0
                best_theta = self.theta.copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Ранняя остановка на эпохе {epoch}")
                    # noinspection PyUnboundLocalVariable
                    self.theta = best_theta
                    break

            if verbose and epoch % 25 == 0:
                print(f"Эпоха {epoch:3d}: Loss = {train_loss:.4f}, R² = {train_r2:.4f}, LR = {lr:.6f}")

    def predict(self, X_):
        return X_.dot(self.theta)

    def get_regularization_type(self):
        """Получение строкового описания типа регуляризации"""
        if self.regularization == 'elasticnet':
            return f"ElasticNet (L1 ratio={self.l1_ratio}, alpha={self.alpha})"
        else:
            return f"{self.regularization.upper()} (alpha={self.alpha})"


def decay_exponential(epoch, initial_lr, decay_rate=0.99):
    """Изменение learning rate по экспоненте"""
    return initial_lr * (decay_rate ** epoch)


def decay_by_step(epoch, initial_lr, drop_every=30, drop_factor=0.5):
    """Изменение learning rate по шагам"""
    return initial_lr * (drop_factor ** (epoch // drop_every))


print('''

####################################################
#  ЭКСПЕРИМЕНТ 1: Сравнение методов регуляризации  #
####################################################
''')

regularization_methods = [
    ('Без регуляризации', 'none', 0.0, 0.5),
    ('L1 (Lasso)', 'l1', 0.01, 0.5),
    ('L2 (Ridge)', 'l2', 0.01, 0.5),
    ('ElasticNet (50/50)', 'elasticnet', 0.01, 0.5),
    ('ElasticNet (80% L1)', 'elasticnet', 0.01, 0.8),
    ('ElasticNet (20% L1)', 'elasticnet', 0.01, 0.2),
]

regularization_results = []

for reg_name, reg_type, alpha, l1_ratio in regularization_methods:
    print(f"\n{reg_name}:")

    model = PolynomialRegressionSGD(
        n_features=X_train.shape[1],
        learning_rate=0.02,
        batch_size=32,
        regularization=reg_type,
        alpha_=alpha,
        l1_ratio_=l1_ratio
    )

    time_start = time.perf_counter()
    model.fit(X_train, y_train, n_epochs=150, verbose=False)
    time_trained = time.perf_counter() - time_start

    y_predicted = model.predict(X_test)
    y_predicted_original = scaler_y.inverse_transform(y_predicted)
    y_test_original = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_original, y_predicted_original)
    mae = mean_absolute_error(y_test_original, y_predicted_original)
    r2 = r2_score(y_test_original, y_predicted_original)

    non_zero_coefs_cnt = np.sum(np.abs(model.theta[1:]) > 1e-4)
    total_coefs = len(model.theta) - 1  # исключаем intercept
    sparsity = 1 - (non_zero_coefs_cnt / total_coefs)

    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"  Время обучения: {time_trained:.4f}с")
    print(f"  Ненулевых коэффициентов: "
          f"{non_zero_coefs_cnt}/{total_coefs} "
          f"({sparsity * 100:.1f}% разреженность)")

    regularization_results.append({
        'method': reg_name,
        'type': reg_type,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'time': time_trained,
        'non_zero_coefs': non_zero_coefs_cnt,
        'sparsity': sparsity,
        'loss_history': model.loss_history,
        'r2_history': model.train_r2_history,
        'theta': model.theta.copy()
    })

print('''

#######################################################
#  ЭКСПЕРИМЕНТ 2: Влияние силы регуляризации (alpha)  #
#######################################################
''')

alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
reg_types = ['l1', 'l2']

results_alpha = {reg_type: [] for reg_type in reg_types}

for reg_type in reg_types:
    print(f"\n{reg_type.upper()} регуляризация:")

    for alpha in alphas:
        model = PolynomialRegressionSGD(
            n_features=X_train.shape[1],
            learning_rate=0.02,
            batch_size=32,
            regularization=reg_type,
            alpha_=alpha
        )

        model.fit(X_train, y_train, n_epochs=100, verbose=False)

        y_predicted = model.predict(X_test)
        y_predicted_original = scaler_y.inverse_transform(y_predicted)

        # noinspection PyUnboundLocalVariable
        mse = mean_squared_error(y_test_original, y_predicted_original)
        r2 = r2_score(y_test_original, y_predicted_original)
        non_zero_coefs_cnt = np.sum(np.abs(model.theta[1:]) > 1e-4)

        print(f"  alpha={alpha:.3f}: MSE={mse:.4f}, R²={r2:.4f}, ненулевых={non_zero_coefs_cnt}")

        results_alpha[reg_type].append({
            'alpha': alpha,
            'mse': mse,
            'r2': r2,
            'non_zero_coefs': non_zero_coefs_cnt,
            'theta_norm': np.linalg.norm(model.theta[1:])  # норма весов (без intercept)
        })

print('''

#################################################################
#  ЭКСПЕРИМЕНТ 3: Сравнение с готовыми реализациями из sklearn  #
#################################################################
''')

sgd_configs = [  # SGDRegressor с разными регуляризациями
    ('SGD (L2)', {'penalty': 'l2', 'alpha': 0.01}),
    ('SGD (L1)', {'penalty': 'l1', 'alpha': 0.01}),
    ('SGD (ElasticNet)', {'penalty': 'elasticnet', 'alpha': 0.01, 'l1_ratio': 0.5}),
]

results_sklearn = []
for sgd_name, params in sgd_configs:
    print(f"\n{sgd_name}:")

    model = SGDRegressor(
        max_iter=1000,
        tol=1e-3,
        learning_rate='constant',
        eta0=0.01,
        random_state=42,
        **params
    )

    time_start = time.perf_counter()
    # Исключаем столбец единиц (intercept)
    model.fit(X_train[:, 1:], y_train.ravel())
    time_trained = time.perf_counter() - time_start

    y_predicted = model.predict(X_test[:, 1:]).reshape(-1, 1)
    y_predicted_original = scaler_y.inverse_transform(y_predicted)

    mse = mean_squared_error(y_test_original, y_predicted_original)
    mae = mean_absolute_error(y_test_original, y_predicted_original)
    r2 = r2_score(y_test_original, y_predicted_original)

    # Получаем коэффициенты (добавляем intercept)
    coef = model.coef_.reshape(-1, 1)
    intercept = model.intercept_.reshape(1, 1)
    theta = np.vstack([intercept, coef])

    non_zero_coefs_cnt = np.sum(np.abs(coef) > 1e-4)

    print(f"  Время обучения: {time_trained:.4f}с")
    print(f"  Ненулевых коэффициентов: {non_zero_coefs_cnt}/{len(coef)}")

    results_sklearn.append({
        'name': sgd_name,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'time': time_trained,
        'non_zero_coefs': non_zero_coefs_cnt,
        'theta': theta
    })

print("\nElasticNet (sklearn) для сравнения:")
elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)
time_start = time.perf_counter()
elasticnet.fit(X_train[:, 1:], y_train.ravel())
time_elasticnet = time.perf_counter() - time_start

y_predicted_elasticnet = elasticnet.predict(X_test[:, 1:]).reshape(-1, 1)
y_predicted_elasticnet_original = scaler_y.inverse_transform(y_predicted_elasticnet)

mse_elasticnet = mean_squared_error(y_test_original, y_predicted_elasticnet_original)
r2_elasticnet = r2_score(y_test_original, y_predicted_elasticnet_original)
non_zero_coefs_elasticnet = np.sum(np.abs(elasticnet.coef_) > 1e-4)

print(f"  MSE: {mse_elasticnet:.4f}, R²: {r2_elasticnet:.4f}")
print(f"  Время обучения: {time_elasticnet:.4f}с")
print(f"  Ненулевых коэффициентов: {non_zero_coefs_elasticnet}/{len(elasticnet.coef_)}")

results_sklearn.append({
    'name': 'ElasticNet (sklearn)',
    'mse': mse_elasticnet,
    'mae': mean_absolute_error(y_test_original, y_predicted_elasticnet_original),
    'r2': r2_elasticnet,
    'time': time_elasticnet,
    'non_zero_coefs': non_zero_coefs_elasticnet,
    'theta': np.vstack([elasticnet.intercept_.reshape(1, 1), elasticnet.coef_.reshape(-1, 1)])
})

print('''

####################################################################
#  ЭКСПЕРИМЕНТ 4: Оптимизация ElasticNet с разными стратегиями LR  #
####################################################################
''')

lr_strategies = [
    ('Постоянный LR', None),
    ('Экспоненциальный затухание', lambda epoch, lr: decay_exponential(epoch, lr, decay_rate=0.98)),
    ('Ступенчатый', lambda epoch, lr: decay_by_step(epoch, lr, drop_every=20, drop_factor=0.5)),
]

results_lr_elasticnet = []

for strategy_name, schedule_func in lr_strategies:
    print(f"\n{strategy_name}:")

    model = PolynomialRegressionSGD(
        n_features=X_train.shape[1],
        learning_rate=0.05,
        batch_size=32,
        regularization='elasticnet',
        alpha_=0.01,
        l1_ratio_=0.5
    )

    time_start = time.perf_counter()
    model.fit(
        X_train, y_train,
        n_epochs=150,
        learning_rate_schedule=schedule_func,
        verbose=False
    )
    time_trained = time.perf_counter() - time_start

    y_predicted = model.predict(X_test)
    y_predicted_original = scaler_y.inverse_transform(y_predicted)

    mse = mean_squared_error(y_test_original, y_predicted_original)
    r2 = r2_score(y_test_original, y_predicted_original)

    print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
    print(f"  Время обучения: {time_trained:.2f}с")

    results_lr_elasticnet.append({
        'strategy': strategy_name,
        'mse': mse,
        'r2': r2,
        'time': time_trained,
        'loss_history': model.loss_history
    })

############################################################################################
#  ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ: Полиномиальная регрессия с регуляризацией: сравнение методов  #
############################################################################################

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

"""  # Схожие результаты, ненаглядно
# 1. Сравнение методов регуляризации (MSE)
plt.figure(num="РегMSE", figsize=(12, 8))
methods = [r['method'] for r in regularization_results]
mse_values = [r['mse'] for r in regularization_results]
#
x_pos = np.arange(len(methods))
bars = plt.bar(x_pos, mse_values, color=colors)
plt.xlabel('Метод регуляризации')
plt.ylabel('MSE (меньше лучше)')
plt.title('Сравнение MSE для разных методов регуляризации')
plt.xticks(x_pos, methods, rotation=45, ha='right')
plt.grid(visible=True, alpha=0.4, axis='y')
#
# Добавление значений на столбцы
for bar, mse in zip(bars, mse_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
            s=f'{mse:.3f}', ha='center', va='bottom', fontsize=9)
#
plt.tight_layout()
plt.show(block=False)

# 2. Сравнение методов регуляризации (R²)
plt.figure(num="РегR2", figsize=(12, 8))
r2_values = [r['r2'] for r in regularization_results]
x_pos = np.arange(len(methods))
bars = plt.bar(x_pos, r2_values, color=colors)
plt.xlabel('Метод регуляризации')
plt.ylabel('R² (больше лучше)')
plt.title('Сравнение R² для разных методов регуляризации')
plt.xticks(x_pos, methods, rotation=45, ha='right')
plt.grid(visible=True, alpha=0.4, axis='y')
#
# Добавление значений на столбцы
for bar, r2 in zip(bars, r2_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
            s=f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
#
plt.tight_layout()
plt.show(block=False)
"""

# 3. Сходимость обучения для разных методов регуляризации
plt.figure(num="СхоОбу", figsize=(12, 7))
for i, result in enumerate(regularization_results[:4]):  # Первые 4 метода
    plt.plot(result['loss_history'][:100], label=result['method'], color=colors[i])
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сходимость обучения')
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# 4. Влияние силы регуляризации на MSE
plt.figure(num="СилMSE", figsize=(12, 7))
for reg_type in reg_types:
    alphas_vals = [r['alpha'] for r in results_alpha[reg_type]]
    mse_vals = [r['mse'] for r in results_alpha[reg_type]]
    plt.plot(alphas_vals, mse_vals, 'o-', label=f'{reg_type.upper()}', linewidth=2)
plt.xlabel('Alpha (сила регуляризации)')
plt.ylabel('MSE')
plt.title('Влияние силы регуляризации на качество')
plt.xscale('log')
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)

"""
# 5. Влияние силы регуляризации на количество ненулевых коэффициентов
plt.figure(num="СилNonZer", figsize=(12, 7))
for reg_type in reg_types:
    alphas_vals = [r['alpha'] for r in results_alpha[reg_type]]
    non_zero = [r['non_zero_coefs'] for r in results_alpha[reg_type]]
    plt.plot(alphas_vals, non_zero, 'o-', label=f'{reg_type.upper()}', linewidth=2)
plt.xlabel('Alpha (сила регуляризации)')
plt.ylabel('Количество ненулевых коэффициентов')
plt.title('Разреженность моделей в зависимости от alpha')
plt.xscale('log')
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)
"""

#######################################################################################
#  Дополнительная визуализация: распределение весов для разных методов регуляризации  #
#######################################################################################

fig2, axes2 = plt.subplots(num="РасВесКоэ", nrows=2, ncols=2, figsize=(15, 12))
fig2.suptitle(t='Распределение весов коэффициентов для разных методов регуляризации', fontsize=16)

# Выбираем 4 интересных метода для визуализации
methods_to_plot = [0, 1, 2, 3]  # Без регуляризации, L1, L2, ElasticNet(50/50)

for idx, method_idx in enumerate(methods_to_plot):
    ax = axes2[idx // 2, idx % 2]
    result = regularization_results[method_idx]
    theta = result['theta'][1:]  # исключаем intercept

    # Гистограмма весов
    ax.hist(theta, bins=int(len(theta) ** 0.5), alpha=0.7, color=colors[method_idx])
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    ax.set_title(f'{result["method"]}    Среднее |вес|: {np.mean(np.abs(theta)):.4f}')
    ax.grid(True, alpha=0.4)
plt.show(block=True)

"""
# Визуализация весов ElasticNet с разными alpha
plt.figure(num="ElaNet")
for alpha_idx, alpha in enumerate([0.001, 0.01, 0.1, 1.0]):
    # Создаем модель L1 с разными alpha
    model = PolynomialRegressionSGD(
        n_features=X_train.shape[1],
        learning_rate=0.02,
        batch_size=32,
        regularization='l1',
        alpha_=alpha
    )
    model.fit(X_train, y_train, n_epochs=100, verbose=False)

    # Считаем количество ненулевых весов
    non_zero = np.sum(np.abs(model.theta[1:]) > 1e-4)
    plt.scatter(alpha, non_zero, s=100, label=f'alpha={alpha}')

    # Добавляем текст с процентом нулевых весов
    total_coefs = len(model.theta) - 1
    zero_pct = 100 * (1 - non_zero / total_coefs)
    plt.text(alpha, non_zero + 5, s=f'{zero_pct:.0f}%', ha='center', fontsize=9)

plt.xlabel('Alpha (сила L1 регуляризации)')
plt.ylabel('Количество ненулевых весов')
plt.title('Влияние L1 регуляризации на разреженность')
plt.xscale('log')
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.show(block=True)
"""

print('''

##########################################
#  ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ  #
##########################################
''')

summary_data = []
for result in regularization_results:
    summary_data.append([
        result['method'],
        f'{result["sparsity"] * 100:.4f}%',
        f'{result["time"]:.4f}с'
    ])

for result in results_sklearn:
    sparsity = 1 - (result['non_zero_coefs'] / (len(result['theta']) - 1))
    summary_data.append([
        result['name'],
        f'{sparsity * 100:.4f}%',
        f'{result["time"]:.4f}с'
    ])

headers = ['Метод', 'Разреженность', 'Время']
print(pd.DataFrame(summary_data, columns=headers).to_string(index=False))

print('''

#################################################
#  АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ ЛУЧШЕЙ МОДЕЛИ  #
#################################################
''')

best_idx = np.argmax([r['r2'] for r in regularization_results])
best_result = regularization_results[best_idx]
print(f"Лучшая модель: {best_result['method']} (R² = {best_result['r2']:.4f})")

feature_names_original = list(datFra.columns[:-1])  # исключаем target
feature_names_polynomial = poly.get_feature_names_out(input_features=feature_names_original)

feature_names_all = ['intercept'] + list(feature_names_polynomial)

weights_df = pd.DataFrame({
    'Признак': feature_names_all,
    'Вес': best_result['theta'].flatten(),
    'Абсолютный_вес': np.abs(best_result['theta'].flatten())
})

weights_sorted = weights_df.sort_values(by='Абсолютный_вес', ascending=False)

print("\nТоп-15 самых важных признаков:")
print(weights_sorted.head(15).to_string(index=False))

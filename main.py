import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import os

# Função para simular dados de sensores
def simular_dados(qtd_amostras=1000):
    dados = {
        'temperatura': np.random.normal(25, 5, qtd_amostras),
        'som': np.random.uniform(30, 100, qtd_amostras),
        'distancia': np.random.uniform(0.2, 5.0, qtd_amostras),
        'gas': np.random.normal(300, 100, qtd_amostras),
        'vibracao': np.random.choice([0, 1], qtd_amostras, p=[0.95, 0.05]),
        'umidade': np.random.uniform(10, 100, qtd_amostras),
        'classe': np.random.choice(['nada', 'vitima', 'perigo'], qtd_amostras, p=[0.5, 0.3, 0.2])
    }
    return pd.DataFrame(dados)

# Gerar dados
df = simular_dados(2000)

# Visualizar as primeiras 10 amostras
print("🔢 Amostras iniciais:")
print(df.head(10))

# Visualizar correlações entre variáveis numéricas
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns=['classe']).corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação dos Sensores")
plt.show()

# Codificação da variável alvo (classe)
label_encoder = LabelEncoder()
df['classe_encoded'] = label_encoder.fit_transform(df['classe'])

# Separar variáveis independentes (X) e alvo (y)
X = df.drop(['classe', 'classe_encoded'], axis=1)
y = df['classe_encoded']

# Normalização dos dados de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding da saída
y_cat = to_categorical(y, num_classes=3)

# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

# Verifica se já existe um modelo salvo
modelo_path = "modelo_robosensor.h5"
if os.path.exists(modelo_path):
    print("📂 Carregando modelo salvo...")
    model = load_model(modelo_path)
else:
    print("⚙️ Treinando novo modelo...")
    model = Sequential([
        Dense(32, input_dim=6, activation='relu'),
        Dropout(0.3),  # Reduz overfitting
        Dense(16, activation='relu'),
        Dropout(0.2),  # Mais regularização
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Salvar modelo treinado
    model.save(modelo_path)
    print("✅ Modelo salvo!")

    # Plotar histórico de acurácia e perda
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Avaliar modelo
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Acurácia no teste: {acc * 100:.2f}%")

# MATRIZ DE CONFUSÃO
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
labels = label_encoder.classes_

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("🔮 Classe Predita")
plt.ylabel("🎯 Classe Real")
plt.title("🧩 Matriz de Confusão")
plt.tight_layout()
plt.show()

# Função de decisão com base na previsão
def simular_decisao(sensor_input):
    sensor_input = scaler.transform([sensor_input])
    pred = model.predict(sensor_input, verbose=0)
    classe_predita = label_encoder.inverse_transform([np.argmax(pred)])[0]
    
    if classe_predita == "vitima":
        return "🚑 Localizar vítima e entregar item"
    elif classe_predita == "perigo":
        return "⚠️ Emitir alerta e fugir da área"
    else:
        return "📡 Continuar mapeando terreno"

# Simular decisões para as 10 primeiras amostras
print("\n🤖 Decisões do robô para as primeiras 10 amostras:")
for i in range(10):
    amostra = df.iloc[i][['temperatura', 'som', 'distancia', 'gas', 'vibracao', 'umidade']].values
    decisao = simular_decisao(amostra)
    classe_real = df.iloc[i]['classe']
    print(f"Amostra {i+1:02}: Real = {classe_real:<7} | Decisão → {decisao}")

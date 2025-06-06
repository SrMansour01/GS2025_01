🤖 RoboSensor Classifier
Este projeto simula dados de sensores ambientais e utiliza uma rede neural para classificar situações como "nada", "vítima" ou "perigo", com base nos dados simulados. Ele pode ser usado como base para sistemas embarcados em robôs de busca e resgate.

📦 Requisitos
Instale as dependências com:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow
```

📋 Funcionalidades
Geração de dados sintéticos simulando sensores (temperatura, som, gás, etc).

Visualização gráfica das correlações entre sensores.

Treinamento ou carregamento de um modelo de rede neural.

Avaliação com matriz de confusão.

Simulação de decisões baseadas nas previsões do modelo.

🧠 Tecnologias utilizadas
- NumPy, Pandas – manipulação de dados

- Matplotlib, Seaborn – visualização

- scikit-learn – pré-processamento e avaliação

- Keras (com TensorFlow) – construção e treino do modelo

🚀 Como executar
Clone o repositório:

```bash
git clone https://github.com/seu-usuario/robosensor-classifier.git
cd robosensor-classifier
```

Execute o script:

```bash
python main.py
```
O código irá:

- Gerar um dataset com 2000 amostras.

- Treinar (ou carregar) uma rede neural.

- Exibir os gráficos de desempenho e matriz de confusão.

- Simular decisões baseadas nas primeiras 10 amostras.

📊 Exemplo de saída
```sql
Amostra 01: Real = nada    | Decisão → 📡 Continuar mapeando terreno
Amostra 02: Real = vitima  | Decisão → 🚑 Localizar vítima e entregar item
Amostra 03: Real = perigo  | Decisão → ⚠️ Emitir alerta e fugir da área
...

```

📁 Estrutura do projeto
```bash
├── main.py   # Código principal
├── modelo_robosensor.h5       # (Gerado após o treinamento)
└── README.md                  # Este arquivo
```
📌 Observações
O modelo é salvo como modelo_robosensor.h5. Na próxima execução, ele será carregado em vez de re-treinado.

A função simular_decisao() pode ser adaptada para uso em um robô real com sensores físicos.

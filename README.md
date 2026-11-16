# Classificação de desfechos de insuficiência cardíaca com Machine Learning

Projeto acadêmico em Python que compara três algoritmos supervisionados — **KNN, SVM e MLP** — usando o dataset *Heart Failure Clinical Records*.

> **Aviso:** este projeto é exclusivamente educacional e experimental. Não é uma ferramenta para diagnóstico, prognóstico ou tomada de decisão clínica.

## Objetivo

Investigar o desempenho de diferentes classificadores para a variável `DEATH_EVENT`, com comparação de acurácia, precisão, recall e F1-score, seleção de hiperparâmetros e curvas de aprendizado.

## Técnicas utilizadas

- **KNN:** classificação baseada na proximidade entre observações.
- **SVM:** classificação por margens e kernels.
- **MLP:** rede neural artificial multicamadas.
- **`Pipeline` + `StandardScaler`:** normalização realizada dentro de cada divisão da validação cruzada, reduzindo o risco de vazamento de dados.
- **`GridSearchCV`:** seleção de hiperparâmetros com validação cruzada estratificada de cinco divisões e otimização de F1-score.
- **Holdout:** separação estratificada em 75% para treino e 25% para teste, com `random_state=42`.

## Tecnologias

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Tkinter

## Como executar

Requer Python 3.10 ou superior e Tkinter instalado.

Baixe ou clone este projeto após criar o repositório no GitHub. Entre na pasta do projeto e crie o ambiente virtual:

```bash
python -m venv .venv
```

No Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

No Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Clique em **Treinar e comparar KNN, SVM e MLP**. A aplicação mostra as métricas em uma tabela e salva os gráficos de comparação, as curvas de aprendizado e `resultados.csv` na pasta `graficos/`. A busca de hiperparâmetros, especialmente da MLP, pode demorar alguns minutos conforme o computador.

## Dataset

O arquivo `heart_failure_clinical_records_dataset.csv` contém 299 registros e 13 colunas, com a variável-alvo `DEATH_EVENT`. O atributo `time` representa o tempo de acompanhamento e pode carregar informação não disponível no momento de uma avaliação inicial. Por isso, os resultados deste experimento **não devem ser interpretados como previsão clínica no momento da admissão**.

O dataset é conhecido como *Heart Failure Clinical Records*, de Chicco e Jurman (2020). Verifique licença e atribuição apropriadas caso reutilize os dados em outro contexto.

## Testes

```bash
python -m pytest tests/ -q
```

Para executar os testes, instale também `pytest` (`pip install pytest`).

## Resultados e limitações

As métricas devem ser interpretadas a partir da execução do código; este README não apresenta números de desempenho não verificados. O conjunto é pequeno e as estimativas estão sujeitas a variação. O teste de holdout é reservado para avaliação final, enquanto a busca de hiperparâmetros utiliza apenas o conjunto de treinamento.

As curvas de aprendizado utilizam os parâmetros selecionados no conjunto de treinamento e servem como exploração visual; **não representam uma estimativa totalmente independente do processo de seleção de modelos**.

## Estrutura

```text
heart-failure-ml/
├── main.py
├── modelos.py
├── heart_failure_clinical_records_dataset.csv
├── tests/
├── graficos/ (gerada na execução)
├── requirements.txt
├── .gitignore
└── README.md
```

## Autora

**Ana Flávia Bitu** — estudante de Ciência da Computação.

[GitHub](https://github.com/anaflaviabitu)

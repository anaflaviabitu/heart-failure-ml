from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "heart_failure_clinical_records_dataset.csv"
GRAFICOS = BASE_DIR / "graficos"


def configuracoes():
    return {
        "KNN": (
            KNeighborsClassifier(),
            {"modelo__n_neighbors": [3, 5, 7, 9, 11], "modelo__weights": ["uniform", "distance"], "modelo__metric": ["euclidean", "manhattan"]},
        ),
        "SVM": (
            SVC(),
            {"modelo__kernel": ["linear", "rbf", "poly"], "modelo__C": [0.1, 1, 10], "modelo__degree": [2, 3]},
        ),
        "MLP": (
            MLPClassifier(random_state=42, max_iter=600, early_stopping=True),
            {"modelo__hidden_layer_sizes": [(50,), (100,)], "modelo__activation": ["relu", "tanh"], "modelo__alpha": [0.0001, 0.001]},
        ),
    }


def gerar_graficos(resultados, destino=GRAFICOS):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for coluna, arquivo in [("Acurácia", "grafico_acuracia.png"), ("Precisão", "grafico_precisao.png"), ("Recall", "grafico_recall.png"), ("F1 Score", "grafico_f1score.png")]:
        figura, eixo = plt.subplots(figsize=(9, 5))
        eixo.bar(resultados["Modelo"], resultados[coluna])
        eixo.set_title(f"Comparação de {coluna}")
        eixo.set_ylabel(coluna)
        eixo.set_ylim(0, 1)
        eixo.tick_params(axis="x", rotation=15)
        figura.tight_layout()
        figura.savefig(destino / arquivo, dpi=180)
        plt.close(figura)


def gerar_curva(nome, modelo, X, y, cv, destino=GRAFICOS):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    tamanhos, treino, validacao = learning_curve(
        modelo,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 5),
        scoring="accuracy",
        error_score=np.nan,
    )
    figura, eixo = plt.subplots(figsize=(8, 5))
    eixo.plot(tamanhos, np.nanmean(treino, axis=1), marker="o", label="Treino")
    eixo.plot(tamanhos, np.nanmean(validacao, axis=1), marker="o", label="Validação")
    eixo.set_title(f"Curva de aprendizado — {nome}")
    eixo.set_xlabel("Número de amostras")
    eixo.set_ylabel("Acurácia")
    eixo.set_ylim(0, 1)
    eixo.legend()
    eixo.grid(alpha=0.2)
    figura.tight_layout()
    figura.savefig(destino / f"learning_curve_{nome}.png", dpi=180)
    plt.close(figura)


def executar(dataset=DATASET, destino=GRAFICOS, gerar_imagens=True, modelos=None, cv_folds=5):
    df = pd.read_csv(dataset)
    if "DEATH_EVENT" not in df.columns:
        raise ValueError("Dataset sem a coluna DEATH_EVENT")
    if df.isnull().any().any():
        raise ValueError("O dataset contém valores ausentes; é necessário tratar esses dados antes do treinamento")
    X = df.drop(columns="DEATH_EVENT")
    y = df["DEATH_EVENT"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    resultados = []
    detalhes = {}
    configurados = modelos or configuracoes()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for nome, (classificador, parametros) in configurados.items():
            pipeline = Pipeline([("normalizacao", StandardScaler()), ("modelo", classificador)])
            busca = GridSearchCV(pipeline, parametros, cv=cv, scoring="f1", n_jobs=1, refit=True)
            busca.fit(X_treino, y_treino)
            previsao = busca.predict(X_teste)
            resultados.append({
                "Modelo": nome,
                "Acurácia": accuracy_score(y_teste, previsao),
                "Precisão": precision_score(y_teste, previsao, zero_division=0),
                "Recall": recall_score(y_teste, previsao, zero_division=0),
                "F1 Score": f1_score(y_teste, previsao, zero_division=0),
            })
            detalhes[nome] = {"melhores_parametros": busca.best_params_, "f1_validacao_cruzada": float(busca.best_score_)}
            if gerar_imagens:
                gerar_curva(nome, busca.best_estimator_, X_treino, y_treino, cv, destino)

    tabela = pd.DataFrame(resultados)
    if gerar_imagens:
        gerar_graficos(tabela, destino)
        tabela.to_csv(Path(destino) / "resultados.csv", index=False)
    return tabela, detalhes

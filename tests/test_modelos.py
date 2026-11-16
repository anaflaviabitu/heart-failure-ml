from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from modelos import DATASET, executar


def test_dataset_existe():
    assert DATASET.exists()


def test_treinamento_basico():
    tabela, detalhes = executar(
        gerar_imagens=False,
        modelos={"KNN": (KNeighborsClassifier(), {"modelo__n_neighbors": [3, 5]})},
        cv_folds=3,
    )
    assert len(tabela) == 1
    assert 0 <= tabela.loc[0, "F1 Score"] <= 1
    assert "KNN" in detalhes

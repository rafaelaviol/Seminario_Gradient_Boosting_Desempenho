from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from time import perf_counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

app = Flask(__name__, template_folder="aplicacao_web")

# Carrega os modelos, métricas e dados preparados pelo treinar_modelo.py.
dados = joblib.load("modelo_xgboost.joblib")

modelo = dados["modelo"]
modelo_logistico = dados["modelo_logistico"]
codificador = dados["codificador"]
colunas_features = dados["colunas_features"]
historico = dados["historico"]
elo = dados["elo"]
metricas_logistico = dados["metricas_logistico"]
metricas_xgboost = dados["metricas_xgboost"]
configuracao_xgboost = dados["configuracao_xgboost"]
X_train = dados["X_train"]
y_train = dados["y_train"]
X_test = dados["X_test"]
y_test = dados["y_test"]


# Calcula as estatísticas históricas de um time.
def calcular_estatisticas(time):
    jogos = historico.get(time, [])

    if not jogos:
        return {"ppg": 0.0, "gols_marcados": 0.0, "gols_sofridos": 0.0}

    return {
        "ppg": float(np.mean([j["pontos"] for j in jogos])),
        "gols_marcados": float(np.mean([j["gols_marcados"] for j in jogos])),
        "gols_sofridos": float(np.mean([j["gols_sofridos"] for j in jogos]))
    }


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/api/teams", methods=["GET"])
def listar_times():
    times = sorted(set(historico.keys()) | set(elo.keys()))
    return jsonify(times)


@app.route("/api/predict", methods=["POST"])
def prever_resultado():
    dados_requisicao = request.get_json(silent=True) or {}
    mandante = dados_requisicao.get("mandante")
    visitante = dados_requisicao.get("visitante")

    if not mandante or not visitante:
        return jsonify({"erro": "Informe o time mandante e o visitante."}), 400
    if mandante == visitante:
        return jsonify({"erro": "Escolha dois times diferentes."}), 400

    times_disponiveis = set(historico.keys()) | set(elo.keys())
    if mandante not in times_disponiveis:
        return jsonify({"erro": "Time mandante não encontrado."}), 404
    if visitante not in times_disponiveis:
        return jsonify({"erro": "Time visitante não encontrado."}), 404

    stats_mandante = calcular_estatisticas(mandante)
    stats_visitante = calcular_estatisticas(visitante)
    elo_mandante = float(elo.get(mandante, 1500.0))
    elo_visitante = float(elo.get(visitante, 1500.0))

    features = {
        "elo_mandante": elo_mandante,
        "elo_visitante": elo_visitante,
        "diferenca_elo": elo_mandante - elo_visitante,
        "ppg_mandante": stats_mandante["ppg"],
        "ppg_visitante": stats_visitante["ppg"],
        "media_gols_marcados_mandante": stats_mandante["gols_marcados"],
        "media_gols_sofridos_mandante": stats_mandante["gols_sofridos"],
        "media_gols_marcados_visitante": stats_visitante["gols_marcados"],
        "media_gols_sofridos_visitante": stats_visitante["gols_sofridos"]
    }

    entrada = [[features[coluna] for coluna in colunas_features]]
    probabilidades = modelo.predict_proba(entrada)[0]
    indice = int(np.argmax(probabilidades))
    resultado = codificador.inverse_transform([indice])[0]
    probabilidades_por_resultado = {
        classe: float(probabilidades[i])
        for i, classe in enumerate(codificador.classes_)
    }

    return jsonify({
        "mandante": mandante,
        "visitante": visitante,
        "resultado": resultado,
        "probabilidades": probabilidades_por_resultado,
        "estatisticas": {
            "mandante": {"elo": elo_mandante, **stats_mandante},
            "visitante": {"elo": elo_visitante, **stats_visitante}
        }
    })


@app.route("/api/model-comparison", methods=["GET"])
def comparar_modelos():
    """Retorna as métricas já calculadas no treinamento original."""
    return jsonify({
        "regressao_logistica": metricas_logistico,
        "xgboost_original": metricas_xgboost,
        "configuracao_xgboost": configuracao_xgboost,
        "quantidade_teste": int(len(y_test))
    })


@app.route("/api/experiment", methods=["POST"])
def executar_experimento():
    """Treina um XGBoost temporário com os parâmetros enviados pela interface."""
    parametros = request.get_json(silent=True) or {}

    limites = {
        "n_estimators": (50, 600, int),
        "max_depth": (1, 8, int),
        "learning_rate": (0.01, 0.30, float),
        "min_child_weight": (1, 10, float),
        "subsample": (0.5, 1.0, float),
        "reg_lambda": (0.0, 10.0, float)
    }

    valores = {}
    try:
        for nome, (minimo, maximo, tipo) in limites.items():
            valor = tipo(parametros[nome])
            if valor < minimo or valor > maximo:
                return jsonify({"erro": f"O parâmetro {nome} deve estar entre {minimo} e {maximo}."}), 400
            valores[nome] = valor
    except (KeyError, TypeError, ValueError):
        return jsonify({"erro": "Informe todos os hiperparâmetros com valores válidos."}), 400

    modelo_teste = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=valores["n_estimators"],
        max_depth=valores["max_depth"],
        learning_rate=valores["learning_rate"],
        min_child_weight=valores["min_child_weight"],
        subsample=valores["subsample"],
        colsample_bytree=0.8,
        reg_lambda=valores["reg_lambda"],
        eval_metric="mlogloss",
        random_state=42
    )

    inicio_treino = perf_counter()
    modelo_teste.fit(X_train, y_train)
    duracao = perf_counter() - inicio_treino
    previsoes = modelo_teste.predict(X_test)

    resultado = {
        "acuracia": float(accuracy_score(y_test, previsoes)),
        "f1_macro": float(f1_score(y_test, previsoes, average="macro", zero_division=0)),
        "tempo_segundos": float(duracao),
        "matriz_confusao": confusion_matrix(
            y_test, previsoes, labels=np.arange(len(codificador.classes_))
        ).tolist(),
        "classes": [str(classe) for classe in codificador.classes_],
        "parametros": valores
    }

    return jsonify({
        "experimento": resultado,
        "regressao_logistica": metricas_logistico,
        "xgboost_original": metricas_xgboost
    })


if __name__ == "__main__":
    app.run(debug=True)

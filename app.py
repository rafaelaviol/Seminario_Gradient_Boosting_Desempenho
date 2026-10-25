from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder="aplicacao_web")

# Carrega o modelo treinado e os dados históricos
dados = joblib.load("modelo_xgboost.joblib")

modelo = dados["modelo"]
codificador = dados["codificador"]
colunas_features = dados["colunas_features"]
historico = dados["historico"]
elo = dados["elo"]


# Calcula as estatísticas históricas de um time
def calcular_estatisticas(time):
    jogos = historico.get(time, [])

    if not jogos:
        return {
            "ppg": 0.0,
            "gols_marcados": 0.0,
            "gols_sofridos": 0.0
        }

    return {
        "ppg": np.mean([j["pontos"] for j in jogos]),
        "gols_marcados": np.mean(
            [j["gols_marcados"] for j in jogos]
        ),
        "gols_sofridos": np.mean(
            [j["gols_sofridos"] for j in jogos]
        )
    }


# Página principal do site
@app.route("/")
def inicio():
    return render_template("index.html")


# Retorna a lista de times disponíveis
@app.route("/api/teams", methods=["GET"])
def listar_times():
    times = sorted(set(historico.keys()) | set(elo.keys()))

    return jsonify(times)


# Realiza a previsão do resultado
@app.route("/api/predict", methods=["POST"])
def prever_resultado():
    dados_requisicao = request.get_json(silent=True) or {}

    mandante = dados_requisicao.get("mandante")
    visitante = dados_requisicao.get("visitante")

    # Verifica se os times foram informados
    if not mandante or not visitante:
        return jsonify({
            "erro": "Informe o time mandante e o visitante."
        }), 400

    if mandante == visitante:
        return jsonify({
            "erro": "Escolha dois times diferentes."
        }), 400

    times_disponiveis = set(historico.keys()) | set(elo.keys())

    if mandante not in times_disponiveis:
        return jsonify({"erro": "Time mandante não encontrado."}), 404

    if visitante not in times_disponiveis:
        return jsonify({"erro": "Time visitante não encontrado."}), 404

    # Obtém as estatísticas históricas
    stats_mandante = calcular_estatisticas(mandante)
    stats_visitante = calcular_estatisticas(visitante)

    # Obtém as pontuações Elo
    elo_mandante = float(elo.get(mandante, 1500.0))
    elo_visitante = float(elo.get(visitante, 1500.0))

    # Monta as características utilizadas pelo modelo
    features = {
        "elo_mandante": elo_mandante,
        "elo_visitante": elo_visitante,
        "diferenca_elo": elo_mandante - elo_visitante,
        "ppg_mandante": stats_mandante["ppg"],
        "ppg_visitante": stats_visitante["ppg"],
        "media_gols_marcados_mandante":
            stats_mandante["gols_marcados"],
        "media_gols_sofridos_mandante":
            stats_mandante["gols_sofridos"],
        "media_gols_marcados_visitante":
            stats_visitante["gols_marcados"],
        "media_gols_sofridos_visitante":
            stats_visitante["gols_sofridos"]
    }

    # Garante a mesma ordem das colunas usadas no treinamento
    entrada = [[features[coluna] for coluna in colunas_features]]

    # Obtém as probabilidades previstas pelo XGBoost
    probabilidades = modelo.predict_proba(entrada)[0]

    indice = int(np.argmax(probabilidades))
    resultado = codificador.inverse_transform([indice])[0]

    # Organiza as probabilidades por resultado do mandante
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
            "mandante": {
                "elo": elo_mandante,
                **{
                    k: float(v)
                    for k, v in stats_mandante.items()
                }
            },
            "visitante": {
                "elo": elo_visitante,
                **{
                    k: float(v)
                    for k, v in stats_visitante.items()
                }
            }
        }
    })


if __name__ == "__main__":
    app.run(debug=True)

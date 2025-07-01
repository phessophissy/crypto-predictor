from flask import Flask, request, render_template_string
import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime

app = Flask(__name__)

HTML = '''
<!doctype html>
<title>Crypto Predictor</title>
<h1>Predict Future Crypto Price</h1>
<form action="/" method="post">
  Token Symbol (e.g., BTC-USD): <input type="text" name="symbol"><br>
  Days Ahead: <input type="number" name="days"><br>
  <input type="submit" value="Predict">
</form>
{% if prediction %}
  <h2>Predicted Price in {{ days }} days: ${{ prediction }}</h2>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    days = 0
    if request.method == "POST":
        symbol = request.form["symbol"]
        days = int(request.form["days"])
        data = yf.download(symbol, period="1y", interval="1d")
        data = data.reset_index()
        data["Days"] = (data["Date"] - data["Date"].min()).dt.days
        X = data[["Days"]]
        y = data["Close"]
        model = LinearRegression().fit(X, y)
        future_day = X["Days"].max() + days
        prediction = round(float(model.predict([[future_day]])), 2)
    return render_template_string(HTML, prediction=prediction, days=days)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

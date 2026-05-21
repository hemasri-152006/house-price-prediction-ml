from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# DATA
data = {
    'area': [1000,1500,1800,2400,3000],
    'bedrooms': [2,3,3,4,5],
    'price': [300000,450000,500000,650000,800000]
}

df = pd.DataFrame(data)
X = df[['area','bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    user_area = None
    user_price = None
    location = None

    if request.method == 'POST':
        user_area = float(request.form['area'])
        bedrooms = float(request.form['bedrooms'])
        location = request.form['location']

        price = model.predict([[user_area, bedrooms]])
        prediction = round(price[0],2)
        user_price = prediction

    return render_template("index.html",
                           prediction=prediction,
                           location=location,
                           area=list(df['area']),
                           price=list(df['price']),
                           user_area=user_area,
                           user_price=user_price)

if __name__ == '__main__':
    app.run(debug=True)

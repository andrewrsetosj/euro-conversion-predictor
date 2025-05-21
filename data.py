# Andrew Seto, arseto@usc.edu 
# ITP 216, Fall 2024
# Section: 32046 
# Final Project
# Description:
# Program allows the user to select between four different global currencies.
# The user can then see a graph representing how the conversion rate between selected currency and the euro has changed from 2000 to 2024. 
# The user can see the predicted monthly conversion rates. 

import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for
import os
import sqlite3 as sl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
csv_file_path = "euro_conversions.csv"
db = "currency_conversions.db"

# # Download latest version from kagglehub
# import kagglehub
# path = kagglehub.dataset_download("lsind18/euro-exchange-daily-rates-19992020")
# print("Path to dataset files:", path)


# Returns to the home page
@app.route("/")
def home():
    return render_template("home.html", message="Welcome!", currency=currency_dictionary())


# Returns to the plot page
@app.route("/action/comparecurrency", methods=["POST", "GET"])
def compare_currency():
    currency = request.form.get("currency")
    session["currency"] = currency
    if not currency:
        return render_template("home.html", message="Error: You must select a currency!", currency=currency_dictionary())
    
    plot_path = plot_conversion_rate(currency)
    return render_template("plot.html", plot_image=plot_path, currency=currency)


# Returns to the prediction page
@app.route("/mlpredictions", methods=["POST", "GET"])
def make_prediction():
    currency = session.get("currency")
    month = request.form.get("month")
    plot_mse = plot_future_conversion_rates(currency, month)
    return render_template("prediction.html", plot_image=plot_mse[1], currency=currency, mse=plot_mse[0])


# Returns to the home page
@app.route("/goback", methods=["POST", "GET"])
def goback():
    session.clear()
    return redirect(url_for('home'))


# Returns to the plot page
@app.route("/monthselection", methods=["POST", "GET"])
def month_selection():
    plot_path = plot_conversion_rate(currency = session.get("currency"))
    return render_template("plot.html", plot_image=plot_path, currency = session.get("currency"))


# Creates a dictionary of the currency codes and currency names
def currency_dictionary() -> dict[str, str]:
    currencyDict = {}
    currencyDict["USD"] = "US Dollar"
    currencyDict["GBP"] = "Great British Pound Sterling"
    currencyDict["JPY"] = "Japanese Yen"
    currencyDict["AUD"] = "Australian Dollar"
    return currencyDict


# Adding a date to the database
def db_add_date(year: str, month: str, day: str, USD_rate: int, GBP_rate: int, JPY_rate: int, AUD_rate: int):
    conn = sl.connect(db)
    curs = conn.cursor()
    
    # Create the globalcurrencies tables if it doesn't exist
    stmt1 = "CREATE TABLE IF NOT EXISTS globalcurrencies ('year', 'month', 'day', 'USD', 'GBP', 'JPY', 'AUD');"
    curs.execute(stmt1)

    # Adding the day's currency info to the database
    v = (year, month, day, USD_rate, GBP_rate, JPY_rate, AUD_rate)
    stmt2 = "INSERT OR IGNORE INTO globalcurrencies (year, month, day, USD, GBP, JPY, AUD) VALUES (?, ?, ?, ?, ?, ?, ?);"
    curs.execute(stmt2, v)
    conn.commit()
    conn.close()


# Populating the database with the csv file
def db_populate(csv_file_path: str):
    # Read the csv file through pandas and make a dataframe
    df = pd.read_csv(csv_file_path)

    # Going through every row in the dataframe
    for index, row in df.iterrows():
        # Skipping empty rows (if USD is empty, then all are empty)
        if row['[US dollar ]'] == "-":
            continue

        # Extracting desired values from each row
        year = row['Day'][0:4]
        month = row['Day'][5:7]
        day = row['Day'][8:10]
        USD_rate = float(row['[US dollar ]'])
        GBP_rate = float(row['[UK pound sterling ]'])
        JPY_rate = float(row['[Japanese yen ]'])
        AUD_rate = float(row['[Australian dollar ]'])

        # Adding each day to the database
        db_add_date(year, month, day, USD_rate, GBP_rate, JPY_rate, AUD_rate)



# Plots the conversion rate for a given currency from 2000 to 2024
def plot_conversion_rate(currency: str) -> str:
    # Query the database
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt = stmt = "SELECT year, " + currency + " FROM globalcurrencies;"
    data = curs.execute(stmt)
    year_rate_dict = {}
    for row in data:
        year = row[0]
        rate = row[1]
        if year not in year_rate_dict:
            year_rate_dict[year] = []
        year_rate_dict[year].append(rate)
    conn.close()

    # Calculating averages
    years = []
    avg_rates = []
    for year, rates in year_rate_dict.items():
        avg_rate = sum(rates) / len(rates)
        years.append(int(year))
        avg_rates.append(avg_rate)

    # Processing currency ID
    currency_mapping = {
        'USD': 'Quantity of US Dollar per Euro from 2000 - 2024', 
        'GBP': 'Quantity of UK Pound Sterling per Euro from 2000 - 2024',
        'AUD': 'Quantity of Australian Dollar per Euro from 2000 - 2024', 
        'JPY': 'Quantity of Japanese Yen per Euro from 2000 - 2024',
    }
    title = currency_mapping[currency]

    # Plot the data
    plt.plot(years, avg_rates, label="Real Conversion Rate")
    plt.title(title)
    plt.xlabel("Year")
    plt.xticks(range(2000, 2026, 5))
    plt.ylabel(currency + " per Euro")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plot_path = currency + "_conversion_plot.png"
    plt.savefig("static/" + plot_path)
    plt.close()
    return plot_path


# Predicts the future conversion rates for a given month and currency in 2025
def plot_future_conversion_rates(currency: str, month:str) -> list[int, str]:
    # Query database
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt = stmt = "SELECT year, month, day, " + currency + " FROM globalcurrencies;"
    data = curs.execute(stmt)
    date = []
    conversion = []
    for row in data:
        year, month_, day, rate = row
        date.append([int(year), int(month_), int(day)])
        conversion.append(rate)
    conn.close()

    # Convert into np arrays
    x = np.array(date)
    y = np.array(conversion)

    # Split 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # Scale
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(x_train, y_train)

    # Process month key
    month_mapping = {
        '01': (31, "January"),
        '02': (28, "February"),
        '03': (31, "March"),
        '04': (30, "April"),
        '05': (31, "May"),
        '06': (30, "June"),
        '07': (31, "July"),
        '08': (31, "August"),
        '09': (30, "September"),
        '10': (31, "October"),
        '11': (30, "November"),
        '12': (31, "December")
    }
    tuple_data = month_mapping[month]
    num_days = tuple_data[0]
    month_name = tuple_data[1]

    # Test
    predictions = knn.predict(x_test)
    mse = mean_squared_error(y_test, predictions)

    # Predict
    dates_2025 = []
    for day in range(1, num_days + 1):
        dates_2025.append([2025, int(month), day])
    x_2025 = np.array(dates_2025)
    predictions_2025 = knn.predict(x_2025)

    # Create the graph
    plt.plot(range(1, num_days + 1), predictions_2025, label="Predicted Conversion Rate")
    plt.title("Predicted Conversion Rates for " + month_name + " of 2025")
    plt.xlabel("Day")
    plt.ylabel(currency + " per Euro")
    plt.legend()
    plot_path = currency + "_predicted_conversion_2025.png"
    plt.savefig("static/" + plot_path)
    plt.close()
    return[mse, plot_path]
    

# Runs the app 
if __name__ == "__main__":
    db_populate(csv_file_path)
    app.secret_key = os.urandom(12)
    app.run(debug=True)

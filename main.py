import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:,.2f}'.format

data = pd.read_csv('data/boston.csv', index_col=0)

def run_challenge(description, func):
    print(f"\nChallenge: {description}")
    input("Press ENTER to see the result...\n")
    func()


def challenge_shape_and_info():
    print(f"Shape of data: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("First 5 rows:")
    print(data.head())
    print("Last 5 rows:")
    print(data.tail())
    print("Counts per column:")
    print(data.count())
    print("\nInfo:")
    data.info()
    print(f"\nAny NaN values? {data.isna().values.any()}")
    print(f"Any duplicates? {data.duplicated().values.any()}")


def challenge_descriptive_stats():
    print(data.describe())


def challenge_displots():
    sns.displot(data['PRICE'], bins=50, aspect=2, kde=True, color='#2196f3')
    plt.title(f'1970s Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}')
    plt.xlabel('Price in 000s')
    plt.ylabel('Nr. of Homes')
    plt.show()

    sns.displot(data.DIS, bins=50, aspect=2, kde=True, color='darkblue')
    plt.title(f'Distance to Employment Centres. Average: {(data.DIS.mean()):.2}')
    plt.xlabel('Weighted Distance to 5 Boston Employment Centres')
    plt.ylabel('Nr. of Homes')
    plt.show()

    sns.displot(data.RM, aspect=2, kde=True, color='#00796b')
    plt.title(f'Distribution of Rooms in Boston. Average: {data.RM.mean():.2}')
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Nr. of Homes')
    plt.show()

    plt.figure(figsize=(10, 5), dpi=200)
    plt.hist(data['RAD'], bins=24, ec='black', color='#7b1fa2', rwidth=0.5)
    plt.xlabel('Accessibility to Highways')
    plt.ylabel('Nr. of Houses')
    plt.show()


def challenge_plotly_bar_CHAS():
    river_access = data['CHAS'].value_counts()
    bar = px.bar(
        x=['No', 'Yes'],
        y=river_access.values,
        color=river_access.values,
        color_continuous_scale=px.colors.sequential.haline,
        title='Next to Charles River?'
    )
    bar.update_layout(
        xaxis_title='Property Located Next to the River?',
        yaxis_title='Number of Homes',
        coloraxis_showscale=False
    )
    bar.show()


def challenge_pairplot():
    sns.pairplot(data)
    plt.show()


def challenge_jointplots():
    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data['DIS'], y=data['NOX'], height=8, kind='scatter', color='deeppink', joint_kws={'alpha': 0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.NOX, y=data.INDUS, height=7, color='darkgreen', joint_kws={'alpha': 0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data['LSTAT'], y=data['RM'], height=7, color='orange', joint_kws={'alpha': 0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.LSTAT, y=data.PRICE, height=7, color='crimson', joint_kws={'alpha': 0.5})
    plt.show()

    with sns.axes_style('whitegrid'):
        sns.jointplot(x=data.RM, y=data.PRICE, height=7, color='darkblue', joint_kws={'alpha': 0.5})
    plt.show()


def challenge_train_test_split():
    target = data['PRICE']
    features = data.drop('PRICE', axis=1)

    global X_train, X_test, y_train, y_test  # Make available for next challenges
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=10
    )

    train_pct = 100 * len(X_train) / len(features)
    test_pct = 100 * X_test.shape[0] / features.shape[0]
    print(f'Training data is {train_pct:.3}% of the total data.')
    print(f'Test data makes up the remaining {test_pct:.3}%.')


def challenge_multivariable_regression():
    global regr, rsquared  # Keep for later use
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    rsquared = regr.score(X_train, y_train)
    print(f'Training data r-squared: {rsquared:.2}')


def challenge_regression_coefficients():
    regr_coef = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coefficient'])
    print(regr_coef)

    premium = regr_coef.loc['RM'].values[0] * 1000
    print(f'The price premium for having an extra room is ${premium:.5}')


def challenge_regression_residuals():
    predicted_vals = regr.predict(X_train)
    residuals = (y_train - predicted_vals)

    plt.figure(dpi=100)
    plt.scatter(x=y_train, y=predicted_vals, c='indigo', alpha=0.6)
    plt.plot(y_train, y_train, color='cyan')
    plt.title('Actual vs Predicted Prices: $y_i$ vs $\\hat y_i$', fontsize=17)
    plt.xlabel('Actual prices 000s $y_i$', fontsize=14)
    plt.ylabel('Predicted prices 000s $\\hat y_i$', fontsize=14)
    plt.show()

    plt.figure(dpi=100)
    plt.scatter(x=predicted_vals, y=residuals, c='indigo', alpha=0.6)
    plt.title('Residuals vs Predicted Values', fontsize=17)
    plt.xlabel('Predicted Prices $\\hat y_i$', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.show()

    resid_mean = round(residuals.mean(), 2)
    resid_skew = round(residuals.skew(), 2)

    sns.displot(residuals, kde=True, color='indigo')
    plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
    plt.show()


def challenge_log_transform_analysis():
    tgt_skew = data['PRICE'].skew()
    sns.displot(data['PRICE'], kde=True, color='green')
    plt.title(f'Normal Prices. Skew is {tgt_skew:.3}')
    plt.show()

    y_log = np.log(data['PRICE'])
    sns.displot(y_log, kde=True)
    plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
    plt.show()

    plt.figure(dpi=150)
    plt.scatter(data.PRICE, np.log(data.PRICE))
    plt.title('Mapping the Original Price to a Log Price')
    plt.ylabel('Log Price')
    plt.xlabel('Actual $ Price in 000s')
    plt.show()

    return y_log


def challenge_log_regression(y_log):
    global log_regr, log_rsquared, log_predictions, log_residuals, log_y_train, log_y_test
    features = data.drop('PRICE', axis=1)

    X_train, X_test, log_y_train, log_y_test = train_test_split(
        features,
        y_log,
        test_size=0.2,
        random_state=10
    )

    log_regr = LinearRegression()
    log_regr.fit(X_train, log_y_train)
    log_rsquared = log_regr.score(X_train, log_y_train)
    log_predictions = log_regr.predict(X_train)
    log_residuals = log_y_train - log_predictions

    print(f'Training data r-squared: {log_rsquared:.2}')


def challenge_log_regression_coefficients():
    df_coef = pd.DataFrame(data=log_regr.coef_, index=X_train.columns, columns=['Coefficient'])
    print(df_coef)


def challenge_compare_regression_plots():
    # Plot log price regression
    plt.figure(dpi=100)
    plt.scatter(x=log_y_train, y=log_predictions, c='navy', alpha=0.6)
    plt.plot(log_y_train, log_y_train, color='cyan')
    plt.title(f'Actual vs Predicted Log Prices: $y_i$ vs $\\hat y_i$ (R²={log_rsquared:.2})', fontsize=17)
    plt.xlabel('Actual Log Prices $y_i$', fontsize=14)
    plt.ylabel('Predicted Log Prices $\\hat y_i$', fontsize=14)
    plt.show()

    # Plot original price regression
    plt.figure(dpi=100)
    plt.scatter(x=y_train, y=regr.predict(X_train), c='indigo', alpha=0.6)
    plt.plot(y_train, y_train, color='cyan')
    plt.title(f'Original Actual vs Predicted Prices: $y_i$ vs $\\hat y_i$ (R²={rsquared:.3})', fontsize=17)
    plt.xlabel('Actual Prices 000s $y_i$', fontsize=14)
    plt.ylabel('Predicted Prices 000s $\\hat y_i$', fontsize=14)
    plt.show()

    # Residuals vs Predicted values (log prices)
    plt.figure(dpi=100)
    plt.scatter(x=log_predictions, y=log_residuals, c='navy', alpha=0.6)
    plt.title('Residuals vs Fitted Values for Log Prices', fontsize=17)
    plt.xlabel('Predicted Log Prices $\\hat y_i$', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.show()

    # Residuals vs Predicted values (original prices)
    plt.figure(dpi=100)
    residuals = y_train - regr.predict(X_train)
    plt.scatter(x=regr.predict(X_train), y=residuals, c='indigo', alpha=0.6)
    plt.title('Residuals vs Predicted Values (Original Prices)', fontsize=17)
    plt.xlabel('Predicted Prices $\\hat y_i$', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.show()


# ------------------ Run Challenges ------------------

run_challenge("Shape, Info, and Data Integrity Checks", challenge_shape_and_info)

run_challenge("Descriptive Statistics", challenge_descriptive_stats)

run_challenge("Displots for PRICE, DIS, RM, RAD", challenge_displots)

run_challenge("Bar Chart of CHAS (River Access) with Plotly", challenge_plotly_bar_CHAS)

run_challenge("Pairplot of Dataset", challenge_pairplot)

run_challenge("Jointplots for Various Variable Relationships", challenge_jointplots)

run_challenge("Train/Test Split of Dataset", challenge_train_test_split)

run_challenge("Multivariable Linear Regression", challenge_multivariable_regression)

run_challenge("Regression Coefficients", challenge_regression_coefficients)

run_challenge("Regression Residuals & Diagnostics", challenge_regression_residuals)

run_challenge("Regression with Log-Transformed Target", lambda: challenge_log_regression(challenge_log_transform_analysis()))

run_challenge("Coefficients for Log-Transformed Regression", challenge_log_regression_coefficients)

run_challenge("Comparison of Regression Plots", challenge_compare_regression_plots)



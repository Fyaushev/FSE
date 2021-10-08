import numpy as np
import useful_package as up

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    X = np.linspace(1, 10, 500)
    y_poly = up.polynom_3(X)
    y_hyp = up.hyperbola(X)

    x_poly_train, x_poly_test, y_poly_train, y_poly_test = train_test_split(X, y_poly, test_size=0.33, random_state=42)
    x_hyp_train, x_hyp_test, y_hyp_train, y_hyp_test = train_test_split(X, y_hyp, test_size=0.33, random_state=42)

    regr_poly = RandomForestRegressor(max_depth=5, random_state=0)
    regr_poly.fit(x_poly_train.reshape(-1, 1), y_poly_train)

    regr_hyp = RandomForestRegressor(max_depth=5, random_state=0)
    regr_hyp.fit(x_hyp_train.reshape(-1, 1), y_hyp_train)

    mse_poly = ((regr_poly.predict(x_poly_test.reshape(-1, 1)) - y_poly_test)**2).mean()
    mse_hyp = ((regr_hyp.predict(x_hyp_test.reshape(-1, 1)) - y_hyp_test)**2).mean()


    print("MSE for polynom_3: ", mse_poly)
    print("MSE for heperbola: ", mse_hyp)
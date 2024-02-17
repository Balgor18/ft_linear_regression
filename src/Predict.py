import sys
import style
from typing import Tuple
from sklearn.metrics import r2_score
import pandas as pd

def predict(theta : Tuple[float], mileage : int) -> int:
    """Predict function

    Args:
        theta (Tuple[float]): Theta for predict the pric
        mileage (int): mileage of a cars

    Returns:
        int: The price of the cars compared with the mileage
    """
    assert isinstance(mileage, int), "mileage must be an integer"
    assert mileage > 0, "mileage must be greater than 0"

    price = theta[0] + (theta[1] * mileage)
    return price

def _add_data_to_array(file : str, array_elem : int, theta : Tuple[float]) -> Tuple[int]:
    """A data to an array

    Args:
        file (str): The file where i got the data
        array_elem (int): the number of the elements of the file i got
        theta (Tuple[float]): the theta for the predict function

    Returns:
        Tuple[int]: The array with the data i need for the precision
    """
    values : [int] = []
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:] :
            data = int(line.split(',')[array_elem])
            if array_elem == 0 :
                values.append(predict(theta, data))
            else :
                values.append(data)
    return values

def precision_calculation(theta : Tuple[float]) -> float:
    """Function that calculate the precision score of the model with sklearn

    Args:
        theta (Tuple[float]): Theta need for the predict function

    Returns:
        float: The precision of the model
    """

    values : [int]= []
    supposed_val : [int] = []

    values = _add_data_to_array('data.csv', 0, theta)
    supposed_val = _add_data_to_array('data.csv', 1, theta)

    print(style.green('Precision score: ', r2_score(supposed_val, values)))

if __name__ == "__main__":

    try:
        theta : Tuple[float] = [0, 0]
        try:
            data = pd.read_csv('theta.csv')
            theta = [float(x) for x in data.values[0]]
        except:
            print("Error: You must train the model first")
            exit(1)
        value = (int(input("Enter mileage: ")))
        print("Estimated price: ", predict(theta, value))
        if (input("precision of the model? (y/n): ") == 'y'):
            precision_calculation(theta)
    except Exception as e:
        print(style.red("Error: ", e), file=sys.stderr)
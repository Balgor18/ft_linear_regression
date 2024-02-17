from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
import sys

FILENAME="data.csv"
KM=0
PRICE=1
class LinearRegression:
    def __init__(self):
        self.theta : Tuple[int, int] = [0 , 0] # Initialisation de theta
        self.denormTheta : Tuple[int, int] = [0, 0] # Init denormalized theta
        self.default : Tuple[int, int] = [0, 0] # Init des data de depart
        self.normal : Tuple[int, int] = [0, 0] # Init des datas apres normalisation

    def _saveTheta(self) :
        new_csv = pd.DataFrame({'theta0': [self.denormTheta[KM]], 'theta1': [self.denormTheta[PRICE]]})
        new_csv.to_csv('theta.csv', index=False)
        print("Theta Save in file theta.csv")

    def normalize(self, dataset : List) -> List:
        """Normalize a data pass in param

        Args:
            dataset (List): List to normalize

        Returns:
            List: Normalize list
        """
        return ([(x - min(dataset)) / (max(dataset) - min(dataset)) for x in dataset])

    def denormalize(self, thetas: Tuple[float, float], dataset: Tuple) -> Tuple[float, float]:
        """Function denormalize will generate theta from denormalized theta

        Args:
            thetas (Tuple[float, float]): theta denormalized
            dataset (Tuple): the normalized dataset

        Returns:
            Tuple[float, float]: The theta dernomalized from the dataset
        """
        delta = [max(value) - min(value) for value in (dataset[0], dataset[1])]
        thetas[1] = thetas[1] * delta[1] / delta[0] 
        thetas[0] = thetas[0] * delta[1] + min(dataset[1])  - thetas[1] * min(dataset[0])
        return thetas

    def train(self, datas, learning_rate = 0.1):
        sum = [0, 0]
        for i in range(len(datas[0])):
            x, y = [row[i] for row in datas]
            estimated_y = self.theta[0] + (self.theta[1] * x)
            sum[0] += (estimated_y - y)
            sum[1] += (estimated_y - y) * x
        return [self.theta[i] - learning_rate * sum[i] / len(datas[i]) for i in range(len(self.theta))]


    def WhileTrain(self):
        while True:
            new = self.train([self.normal[KM], self.normal[PRICE]])
            if new == self.theta:
                break
            self.theta = new
            plt.plot(self.normal[KM], [self.theta[0] + (self.theta[1] * x) for x in self.normal[KM]], alpha=0.1)
        self.denormTheta = self.denormalize(self.theta, [self.default[KM], self.default[PRICE]])
        self._saveTheta()

def file() -> Tuple[List[int], List[int]]:
    """Open the CSV or a error if not found

    Returns:
        Tuple[List[int], List[int]]: The KM and the Price
    """
    try:
        data = pd.read_csv(FILENAME)

        mileage = data['km'].tolist()
        price = data['price'].tolist()

        return (mileage, price)
    except FileNotFoundError:
        print("Le fichier spécifié est introuvable. Veuillez vérifier le chemin ou le nom du fichier.")
        return [], []
    except Exception as e:
        print("Une erreur s'est produite lors de la lecture du fichier CSV :", e)
        return [], []

def main():
    model = LinearRegression()

    model.default[KM], model.default[PRICE] = file()

    model.normal[KM] = model.normalize(model.default[KM])
    model.normal[PRICE] = model.normalize(model.default[PRICE])

    # plt.subplot(2, 2, 1)
    # plt.scatter(model.default[KM], model.default[PRICE])
    # plt.title('Before normalization')

    # plt.subplot(2, 2, 2)
    # plt.scatter(model.normal[KM], model.normal[PRICE])
    # plt.title('After normalization')

    # plt.subplot(2, 2, 3)
    # plt.scatter(model.normal[KM], model.normal[PRICE])
    # plt.title('Training')

    model.WhileTrain()

    # plt.subplot(2, 2, 4)
    # plt.title('Linear regression')
    # plt.scatter(model.default[KM], model.default[PRICE])
    # plt.plot(model.default[KM], [model.denormTheta[0] + (model.denormTheta[1] * x) for x in model.default[KM]], color='red')
    # plt.show()

if __name__ == '__main__':
    main()
from data_prepare.dataloader import DataLoader
from modeling.neuron import Neuron

def main():
    neu = Neuron()
    neu.plot_region(201)

if __name__ == "__main__":
    main()
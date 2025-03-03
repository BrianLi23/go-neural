package main

import (
	"flag"
	"fmt"

	"github.com/BrianLi23/go-neural/neuralnetwork"
)

// Simple function that returns a string
func getGreeting() string {
	return "Welcome to Go!"
}

func generateTargetVector(label int) []float64 {
	targetVector := make([]float64, 10)
	targetVector[label] = 1.0
	return targetVector
}

func main() {

	seed := 42

	// Load data from the file
	// neuralNetwork := neuralnetwork.LoadNeuralNetwork("neuralnetwork.gob")

	// 784 input nodes, 200 hidden nodes, 10 output nodes
	neuralNetwork := neuralnetwork.NewNeuralNetwork(784, 200, 10, 0.01, seed)

	operation := flag.String("operation", "train", "Operation to perform: train or predict") // returns a pointer to a string
	file := flag.String("file", "", "File to use for prediction")
	flag.Parse()

	if *operation == "predict" {
		neuralNetwork = neuralnetwork.LoadNeuralNetwork("neuralnetwork.gob")
		prediction := neuralnetwork.PredictImage(neuralNetwork, *file)
		fmt.Println("Prediction:", prediction)
		return
	} else if *operation == "train" {
		images, labels, err := neuralnetwork.LoadImageDataset("mnist_train")
		if err != nil {
			fmt.Println("Error loading dataset:", err)
			return
		}
		// Split into train/test
		trainImages, trainLabels, testImages, testLabels := neuralnetwork.SplitDataset(images, labels, 0.8, seed)
		for i, image := range trainImages {
			neuralNetwork.Train(image, generateTargetVector(trainLabels[i]))
			if i%1000 == 0 {
				fmt.Println("Training progress:", i, "/", len(trainImages))
			}
		}
		neuralNetwork.Train(trainImages, trainLabels)

		fmt.Println("Testing on", len(testImages), "images...")
		neuralnetwork.SaveNeuralNetwork("neuralnetwork.gob", neuralNetwork)

		return
	} else {
		println("Invalid operation")
		return
	}

}

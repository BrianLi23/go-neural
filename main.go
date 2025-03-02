package main

import (
	"github.com/BrianLi23/go-neural/neuralnetwork"
)

// Simple function that returns a string
func getGreeting() string {
	return "Welcome to Go!"
}

func main() {
	neuralNetwork := neuralnetwork.NewNeuralNetwork()

	neuralNetwork.Train()

}

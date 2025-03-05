package neuralnetwork

import (
	"encoding/gob"
	"fmt"
	"image"
	"image/draw"
	"math/rand/v2"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

// Defining the neural network struct
type NeuralNetwork struct {
	// Number of input, hidden, and output nodes
	inputNodes  int
	hiddenNodes int
	outputNodes int

	// Weights between input and hidden layer
	// These two fields hold a pointer a mat.Dense struct, this is because the matrix can be large and we don't want to copy it
	hiddenWeights *mat.Dense // These values hold a dense matrix type, where most of the elements are non-zero.
	outputWeights *mat.Dense
	learningRate  float64
	seed          int
}

// Holds the type of the neural network, uses a pointer to the struct
// This is because we want to modify the struct in the methods

func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes int, learningRate float64, seed int) *NeuralNetwork {
	// Initialize the neural network with the provided values
	nn := &NeuralNetwork{
		inputNodes:   inputNodes,
		hiddenNodes:  hiddenNodes,
		outputNodes:  outputNodes,
		learningRate: learningRate,
		seed:         seed,
	}

	// Initialize the weights with random values
	// Think about if I have 4 inputs and 3 hidden nodes, I need a 3x4 matrix to hold the weights
	// hiddenActivations = hiddenWeights * inputs
	// (3 × 4) × (4 × 1) = (3 × 1)
	nn.hiddenWeights = mat.NewDense(nn.hiddenNodes, nn.inputNodes, randomArray(nn.hiddenNodes*nn.inputNodes, float64(nn.inputNodes), nn.seed))
	nn.outputWeights = mat.NewDense(nn.outputNodes, nn.hiddenNodes, randomArray(nn.outputNodes*nn.hiddenNodes, float64(nn.hiddenNodes), nn.seed))

	return nn
}

// Training function for the neural network
func (nn *NeuralNetwork) Train(inputs, targets []float64) {

	// Convert input and target arrays to matrices
	// Input data (4 x 1)
	inputsMat := mat.NewDense(len(inputs), 1, inputs)
	hiddenInputs := dot(nn.hiddenWeights, inputsMat) // (3 x 4) * (4 x 1) = (3 x 1)
	hiddenOutputs := apply(sigmoid, hiddenInputs)    // Apply sigmoid actiavtion function
	finalInputs := dot(nn.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	targetsMat := mat.NewDense(len(targets), 1, targets) // Row by column
	outputError := subtract(targetsMat, finalOutputs)
	hiddenErrors := dot(nn.outputWeights.T(), outputError) // Used to find how much each hidden node contributed to the error .T() is the transpose

	// Backpropagation
	nn.outputWeights = add(nn.outputWeights,
		scale(nn.learningRate,
			dot(multiply(outputError, apply(dsigmoid, finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	nn.hiddenWeights = add(nn.hiddenWeights,
		scale(nn.learningRate,
			dot(multiply(hiddenErrors, apply(dsigmoid, hiddenOutputs)),
				inputsMat.T()))).(*mat.Dense) // (*mat.Dense) is a type assertion to convert the result to a *mat.Dense type

}

// Run prediction on the neural network with the given inputs
func (nn *NeuralNetwork) Predict(inputs []float64) mat.Matrix {
	// Convert input array to a matrix and feedforward
	inputsMat := mat.NewDense(len(inputs), 1, inputs)
	hiddenInputs := dot(nn.hiddenWeights, inputsMat)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(nn.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	return finalOutputs
}

func SaveNeuralNetwork(filename string, nn *NeuralNetwork) {
	// Save the neural network to a file
	h, error := os.Create(filename)
	if error != nil {
		fmt.Println("Error saving neural network to file:", error)
		return
	}
	defer h.Close() // Defer means it will execute last when the function ends (closing file)

	// Save the neural network to the file
	encoder := gob.NewEncoder(h)
	if error := encoder.Encode(nn); error != nil {
		fmt.Println("Error encoding neural network:", error)
	}

	fmt.Println("Neural network saved to file:", filename)
	return

}

func LoadNeuralNetwork(filename string) *NeuralNetwork {
	// Load a neural network from a file
	h, error := os.Open(filename)
	if error != nil {
		fmt.Println("Error loading neural network from file:", error)
		return nil
	}
	defer h.Close()

	// Load the neural network from the file
	nn := &NeuralNetwork{}
	decoder := gob.NewDecoder(h)
	if error := decoder.Decode(nn); error != nil {
		fmt.Println("Error decoding neural network:", error)
		return nil
	}

	fmt.Println("Neural network loaded from file:", filename)
	return nn
}

// Function to predict the image
func PredictImage(nn *NeuralNetwork, imagePath string) int {
	// Load image from file
	image, err := LoadImage(imagePath)
	if err != nil {
		fmt.Println("Error loading image:", err)
		return -1
	}

	// Normalize pixel values
	for i, pixel := range image {
		image[i] = pixel / 255.0
	}

	// Run the prediction
	output := nn.Predict(image)

	// Find the index of the highest value in the output
	prediction := argmax(output)

	return prediction

}

func LoadImage(filename string) ([]float64, error) {
	// Load the image from the file
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error loading image file:", err)
		return nil, err
	}
	defer file.Close()

	// Decode the image
	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Println("Error decoding image file:", err)
		return nil, err
	}

	// Convert the image to a grayscale image
	gray := image.NewGray(img.Bounds())
	draw.Draw(gray, img.Bounds(), img, image.Point{}, draw.Src)

	// Convert the image to a pixel array
	bounds := gray.Bounds()
	var pixels []float64
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := gray.GrayAt(x, y)
			pixels = append(pixels, float64(pixel.Y))
		}
	}

	return pixels, nil

}

func LoadImageDataset(directory string) ([][]float64, []string, error) {
	var images [][]float64
	var labels []string

	err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {

		if err != nil {
			fmt.Println("Error reading directory:", err)
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		if filepath.Ext(path) == ".png" || filepath.Ext(path) == ".jpg" || filepath.Ext(path) == ".jpeg" {
			imgData, err := LoadImage(path)
			if err != nil {
				return nil
			}
			images = append(images, imgData)
			labels = append(labels, path) // Use filename as label (or extract from path)
		}
		return nil
	})

	return images, labels, err
}

// Split the dataset into training and testing sets
func SplitDataset(images [][]float64, labels []string, trainSize float64, seed int) ([][]float64, []string, [][]float64, []string) {
	r := rand.NewPCG(uint64(seed), 1)
	n := len(images)
	perm := r.Perm(n)

	// Shuffle dataset deterministically
	shuffledImages := make([][]float64, n)
	shuffledLabels := make([]string, n)

	for i, idx := range perm {
		shuffledImages[i] = images[idx]
		shuffledLabels[i] = labels[idx]
	}

	// Split into training and testing sets
	trainIndex := int(trainSize * float64(n))
	return shuffledImages[:trainIndex], shuffledLabels[:trainIndex], shuffledImages[trainIndex:], shuffledLabels[trainIndex:]
}

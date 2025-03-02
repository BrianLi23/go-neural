package neuralnetwork

import (
	"encoding/gob"
	"fmt"
	"image"
	"image/draw"
	"math"
	"math/rand/v2"
	"os"

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
}

// Holds the type of the neural network, uses a pointer to the struct
// This is because we want to modify the struct in the methods

func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes int, learningRate float64) *NeuralNetwork {
	// Initialize the neural network with the provided values
	nn := &NeuralNetwork{
		inputNodes:   inputNodes,
		hiddenNodes:  hiddenNodes,
		outputNodes:  outputNodes,
		learningRate: learningRate,
	}

	// Initialize the weights with random values
	// Think about if I have 4 inputs and 3 hidden nodes, I need a 3x4 matrix to hold the weights
	// hiddenActivations = hiddenWeights * inputs
	// (3 × 4) × (4 × 1) = (3 × 1)
	nn.hiddenWeights = mat.NewDense(nn.hiddenNodes, nn.inputNodes, randomArray(nn.hiddenNodes*nn.inputNodes, float64(nn.inputNodes)))
	nn.outputWeights = mat.NewDense(nn.outputNodes, nn.hiddenNodes, randomArray(nn.outputNodes*nn.hiddenNodes, float64(nn.hiddenNodes)))

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

// // Generates a random array of the specified size
// func randomArray(size int, v float64) (data []float64) {
// 	dist := distuv.Uniform{
// 		Min: -0.5,
// 		Max: 0.5,
// 		Source: rand.New(rand.NewSource(
// 			time.Now().UnixNano())),
// 	}
// }

// Defining functions
func dot(m, n mat.Matrix) mat.Matrix {
	// Get the dimensions of the input matrices
	mrows, mcols := m.Dims()
	nrows, ncols := n.Dims()

	// Check if the matrices are compatible for dot product
	if mcols != nrows {
		panic("Matrix dimensions are not compatible for dot product")
	}

	// Create a new matrix to store the result
	result := mat.NewDense(mrows, ncols, nil)

	// Perform the dot product
	result.Mul(m, n)

	return result
}

func apply(fn func(int, int, float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	// Apply the function to each element of the matrix
	result.Apply(fn, m)

	return result
}

func scale(scalar float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	// Scale each element of the matrix
	result.Scale(scalar, m)

	return result
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	// Multiply each element of the matrix
	result.MulElem(m, n)

	return result
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	// Add each element of the matrix
	result.Add(m, n)

	return result
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	// Subtract each element of the matrix
	result.Sub(m, n)

	return result
}

func sigmoid(row, column int, x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(row, column int, x float64) float64 {
	return x * (1 - x)
}

func randomArray(size int, v float64) (data []float64) {
	for i := 0; i < size; i++ {
		data = append(data, (rand.Float64()*2-1)*v)
	}
	return
}

func addScalar(m mat.Matrix, scalar float64) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 { return v + scalar }, m) // A callback function is passed to the Apply method
	return result
}

// Used to add a bias node to the matrix, this is used to add a constant value to the input that is scaled when multiplied by the weights
func addBiasNode(m mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	bias := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		bias.Set(i, 0, 1)
	}
	result := mat.NewDense(0, 0, nil)
	result.Augment(m, bias)

	return result
}

// Used to remove the bias node from the matri
func removeBiasNode(m mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	return mat.DenseCopyOf(m).Slice(0, r, 0, 1)
}

// pretty print a Gonum matrix
func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func save(filename string, nn *NeuralNetwork) {
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

func load(filename string) *NeuralNetwork {
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
func predictImage(nn *NeuralNetwork, imagePath string) int {
	// Load image from file
	image, err := loadImage(imagePath)
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

func loadImage(filename string) ([]float64, error) {
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

func argmax(m mat.Matrix) int {
	// Find the index of the highest value in the matrix
	r, _ := m.Dims()
	maxVal := 0.0
	index := 0
	for i := 0; i < r; i++ {
		value := m.At(i, 0)
		if value > maxVal {
			maxVal = value
			index = i
		}
	}

	return index
}

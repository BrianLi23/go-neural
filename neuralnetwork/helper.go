package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

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

// Generates a random array of the specified size
func randomArray(size int, v float64, seed int) (data []float64) {
	// Create a new random source with the provided seed
	r := rand.New(rand.NewSource(int64(seed)))

	// Pre-allocate the slice
	data = make([]float64, size)

	// Fill with random values between -0.5*v and 0.5*v
	for i := 0; i < size; i++ {
		data[i] = (r.Float64() - 0.5) * v
	}

	return data
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

# go-neural
A fully implemented **Neural Network** with **concurrent training**, written in Go.

## 🚀 Features
- **Fully connected neural network** with customizable layers
- **GPU-accelerated support** (if using Gorgonia)
- **Multi-threaded training** for efficiency
- **Supports MNIST dataset for handwritten digit recognition**
- **Batch training & forward propagation**
- **Custom dataset handling (from image folders or CSV)**

---

## 📂 Project Structure
```
go-neural/
│── dataset/                 # Training dataset (e.g., MNIST)
│   ├── 0/                   # Images of digit '0'
│   ├── 1/                   # Images of digit '1'
│   ├── ...                  # More digit categories
│── neuralnetwork/           # Neural network implementation
│   ├── nn.go                # Core neural network logic
│── .gitignore               # Ignored files
│── go.mod                   # Go module dependencies
│── go.sum                   # Dependency checksums
│── main.go                  # Entry point
│── README.md                # Project documentation
```
---

## 🛠️ **Installation**
1. **Clone the repository**
    git clone https://github.com/yourusername/go-neural.git
    cd go-neural

2. **Install dependencies**
    go mod tidy

🎯 Usage
**Run model**
    go run main.go -operation predict -file path/to/image.png
    go run main.go -operation train

    Output:
    Predicted Value: 5

<!-- 📚 How It Works
Loads training images from dataset/
Normalizes pixel values from 0-255 to 0-1
Forward propagation using a fully connected neural network
Backpropagation & weight updates
Predicts digit classes (0-9) from trained data -->

🖥️ Technologies Used
- Go (Golang) - Primary programming language
- Gorgonia (Optional) - GPU support for tensor computation
- Gonum - Matrix operations
- Parallel Processing - Concurrent training

📌 TODOs & Future Work
- Implement dropout for regularization
- Optimize GPU performance
- Add more dataset compatibility
- Improve training accuracy with Adam optimizer








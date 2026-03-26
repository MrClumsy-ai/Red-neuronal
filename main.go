package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Layer struct {
	weights [][]float64
	biases  []float64
	size    int
}

type NeuralNetwork struct {
	layers       []Layer
	learningRate float64
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func randomWeight() float64 {
	return rand.Float64()*2 - 1
}

func newLayer(inputSize, outputSize int) Layer {
	weights := make([][]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = randomWeight()
		}
	}
	biases := make([]float64, outputSize)
	for i := range biases {
		biases[i] = randomWeight()
	}
	return Layer{
		weights: weights,
		biases:  biases,
		size:    outputSize,
	}
}

func newNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, learningRate float64) *NeuralNetwork {
	layers := make([]Layer, 0)
	prevSize := inputSize
	for _, hiddenSize := range hiddenSizes {
		layers = append(layers, newLayer(prevSize, hiddenSize))
		prevSize = hiddenSize
	}
	layers = append(layers, newLayer(prevSize, outputSize))
	return &NeuralNetwork{
		layers:       layers,
		learningRate: learningRate,
	}
}

func (nn *NeuralNetwork) Forward(data *TrainingData) []float64 {
	current := data.input
	for _, layer := range nn.layers {
		next := make([]float64, layer.size)
		for i := range layer.size {
			sum := layer.biases[i]
			for j := range len(current) {
				sum += current[j] * layer.weights[i][j]
			}
			next[i] = sigmoid(sum)
		}
		current = next
	}
	return current
}

func (nn *NeuralNetwork) Backward(data *TrainingData) {
	activations := make([][]float64, len(nn.layers)+1)
	activations[0] = data.input

	for i, layer := range nn.layers {
		current := activations[i]
		next := make([]float64, layer.size)
		for j := range layer.size {
			sum := layer.biases[j]
			for k := range len(current) {
				sum += current[k] * layer.weights[j][k]
			}
			next[j] = sigmoid(sum)
		}
		activations[i+1] = next
	}
	errors := make([][]float64, len(nn.layers))
	output := activations[len(activations)-1]
	errors[len(errors)-1] = make([]float64, len(output))
	for i := range output {
		errors[len(errors)-1][i] = (data.target[i] - output[i]) * sigmoidDerivative(output[i])
	}
	for i := len(nn.layers) - 2; i >= 0; i-- {
		errors[i] = make([]float64, len(activations[i+1]))
		for j := range errors[i] {
			sum := 0.0
			for k := range errors[i+1] {
				sum += errors[i+1][k] * nn.layers[i+1].weights[k][j]
			}
			errors[i][j] = sum * sigmoidDerivative(activations[i+1][j])
		}
	}
	for i, layer := range nn.layers {
		for j := range layer.weights {
			for k := range layer.weights[j] {
				layer.weights[j][k] += nn.learningRate * errors[i][j] * activations[i][k]
			}
			layer.biases[j] += nn.learningRate * errors[i][j]
		}
	}
}

func (nn *NeuralNetwork) Train(data *TrainingData) {
	nn.Backward(data)
}

func (nn *NeuralNetwork) Loss(prediction []float64, data *TrainingData) float64 {
	loss := 0.0
	for i := range prediction {
		diff := prediction[i] - data.target[i]
		loss += diff * diff
	}
	return loss / float64(len(prediction))
}

type TrainingData = struct {
	input  []float64
	target []float64
}

func newTrainingData(input, target []float64) *TrainingData {
	return &TrainingData{
		input,
		target,
	}
}

func main() {
	const inputSize = 2
	hiddenSizes := []int{3}
	const outputSize = 1
	const learningRate = 0.5
	neuralNetwork := newNeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate)
	trainingData := []TrainingData{
		*newTrainingData([]float64{0, 0}, []float64{0}),
		*newTrainingData([]float64{0, 1}, []float64{1}),
		*newTrainingData([]float64{1, 0}, []float64{1}),
		*newTrainingData([]float64{1, 1}, []float64{0}),
	}

	fmt.Println("training nn...")
	epochs := 10000

	for epoch := range epochs {
		totalLoss := 0.0
		for _, data := range trainingData {
			neuralNetwork.Train(&data)
			prediction := neuralNetwork.Forward(&data)
			totalLoss += neuralNetwork.Loss(prediction, &data)
		}
		if epoch%1000 == 0 {
			fmt.Printf("epoch %d, average loss: %.6f\n", epoch, totalLoss/float64(len(trainingData)))
		}
	}
	fmt.Println("\ntesting trained network:")
	for _, data := range trainingData {
		prediction := neuralNetwork.Forward(&data)
		fmt.Printf("input: %v, expected: %v, prediction: %.4f\n", data.input, data.target, prediction[0])
	}
}

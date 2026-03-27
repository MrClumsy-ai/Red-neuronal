package main

import "github.com/MrClumsy-ai/Red-neuronal/nn"

func main() {
	const inputSize = 2
	hiddenSizes := []int{3}
	const outputSize = 1
	const learningRate = 0.5
	neuralNetwork := nn.NewNeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate)
	trainingData := []nn.TrainingData{
		*nn.NewTrainingData([]float64{0, 0}, []float64{0}),
		*nn.NewTrainingData([]float64{0, 1}, []float64{1}),
		*nn.NewTrainingData([]float64{1, 0}, []float64{1}),
		*nn.NewTrainingData([]float64{1, 1}, []float64{0}),
	}
	const epochs = 10000
	neuralNetwork.Train(trainingData, epochs)
	neuralNetwork.Test(trainingData)
}

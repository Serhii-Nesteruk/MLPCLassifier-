package org.example;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class MLP implements Serializable {

    private static final long serialVersionUID = 1L; // Рекомендовано для Serializable

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private float[][] w1;
    private float[] b1;
    private float[][] w2;
    private float[] b2;

    private transient Random rnd = new Random();

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        w1 = new float[inputSize][hiddenSize];
        b1 = new float[hiddenSize];
        w2 = new float[hiddenSize][outputSize];
        b2 = new float[outputSize];

        initWeights(w1);
        initWeights(w2);
    }

    private void initWeights(float[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = (rnd.nextFloat() - 0.5f) * 0.2f;
            }
        }
    }

    public void train(float[][] inputs, float[][] targets, int epochs, float lr) {
        int n = inputs.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float sumLoss = 0f;
            for (int i = 0; i < n; i++) {
                sumLoss += trainOnExample(inputs[i], targets[i], lr);
            }
            float avgLoss = sumLoss / n;
            System.out.println("Epoch " + epoch + " - Loss: " + avgLoss);
        }
    }

    private float[] calculateSoftmax(float[] outputRaw, float maxLogit) {
        float[] output = new float[outputSize];
        float sumExp = 0f;
        for (int k = 0; k < output.length; k++) {
            output[k] = (float) Math.exp(outputRaw[k] - maxLogit);
            sumExp += output[k];
        }
        for (int k = 0; k < output.length; k++) {
            output[k] /= sumExp;
        }
        return output;
    }

    private float findMaxLogit(float[] hidden) {
        float[] outputRaw = new float[outputSize];
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < outputRaw.length; k++) {
            float sum = b2[k];
            for (int j = 0; j < hidden.length; j++) {
                sum += hidden[j] * w2[j][k];
            }
            outputRaw[k] = sum;
            if (sum > maxLogit) {
                maxLogit = sum;
            }
        }

        return maxLogit;
    }

    private float calculateLoss(float[] output, float[] target) {
        float loss = 0f;
        for (int k = 0; k < outputSize; k++) {
            loss -= (float) (target[k] * Math.log(output[k] + 1e-7f));
        }
        return loss;
    }

    private float[] calculateErrors(float[] output, float[] target) {
        float[] dOutput = new float[outputSize];
        for (int k = 0; k < outputSize; k++) {
            dOutput[k] = output[k] - target[k];
        }
        return dOutput;
    }

    private void backward(float[] hidden, float[] dOutput, float[]hiddenRaw, float[] input, float lr) {
        float[] dHidden = new float[hidden.length];
        for (int k = 0; k < dOutput.length; k++) {
            float grad = dOutput[k];
            for (int j = 0; j < hidden.length; j++) {
                w2[j][k] -= lr * grad * hidden[j];
                dHidden[j] += grad * w2[j][k];
            }
            b2[k] -= lr * grad;
        }
        for (int j = 0; j < hiddenSize; j++) {
            if (hiddenRaw[j] <= 0) {
                dHidden[j] = 0;
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            float grad = dHidden[j];
            for (int i = 0; i < inputSize; i++) {
                w1[i][j] -= lr * grad * input[i];
            }
            b1[j] -= lr * grad;
        }
    }

    private float trainOnExample(float[] input, float[] target, float lr) {
        // ---------- Forward -----------
        float[] hiddenRaw = new float[hiddenSize];
        float[] hidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hiddenRaw[j] = sum;
            hidden[j] = relu(sum);
        }

        float maxLogit = findMaxLogit(hidden);
        float[] output = calculateSoftmax(hiddenRaw, maxLogit);
        float loss = calculateLoss(output, target);

        float[] dOutput = calculateErrors(output, target);

        backward(hidden, dOutput, hiddenRaw, input, lr);

        return loss;
    }

    public PredictionResult predict(float[] input) {
        // Forward
        float[] hidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hidden[j] = relu(sum);
        }

        float[] outputRaw = new float[outputSize];
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < outputSize; k++) {
            float sum = b2[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * w2[j][k];
            }
            outputRaw[k] = sum;
            if (sum > maxLogit) {
                maxLogit = sum;
            }
        }

        float[] output = new float[outputSize];
        float sumExp = 0f;
        for (int k = 0; k < outputSize; k++) {
            output[k] = (float) Math.exp(outputRaw[k] - maxLogit);
            sumExp += output[k];
        }
        for (int k = 0; k < outputSize; k++) {
            output[k] /= sumExp;
        }

        // Аргмакс
        int bestIndex = 0;
        float bestVal = output[0];
        for (int k = 1; k < outputSize; k++) {
            if (output[k] > bestVal) {
                bestVal = output[k];
                bestIndex = k;
            }
        }
        return new PredictionResult(bestIndex, bestVal);
    }

    private float relu(float x) {
        return (x > 0) ? x : 0.01f * x;
    }
}



package org.example;

public class PredictionResult {
    public int predictedIndex;
    public float confidence;

    public PredictionResult(int predictedIndex, float confidence) {
        this.predictedIndex = predictedIndex;
        this.confidence = confidence;
    }
}

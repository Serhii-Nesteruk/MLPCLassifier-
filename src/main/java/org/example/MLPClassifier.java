package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class MLPClassifier {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 56;
    private int[][] binaryPixels = new int[GRID][GRID];
    public final String pathToMLPModel = "mlpModel.bin";
    public final String pathToDataset = "dataset.csv";

    private MLP mlpModel;

    public MLPClassifier() {
        mlpModel = loadModel(pathToMLPModel);
    }

    public PredictionResult predict(float[] inputVec) {
        return mlpModel.predict(inputVec);
    }

    public MLPDataset loadTestSample() {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        try {

            List<String> allLines = Files.readAllLines(Paths.get(pathToDataset));
            int totalLines = allLines.size();

            for (int i = totalLines - 100; i < totalLines; i++) {
                String line = allLines.get(i);
                String[] parts = line.split(",");
                if (parts.length != 1 + (GRID * GRID)) {
                    continue;
                }
                String labelStr = parts[0].trim();
                float[] inVec = new float[GRID * GRID];
                for (int j = 0; j < GRID * GRID; j++) {
                    inVec[j] = Float.parseFloat(parts[j + 1]);
                }
                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0 || classIndex >= 3) {
                    continue;
                }
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;
                inputList.add(inVec);
                targetList.add(targetVec);
            }
            return new MLPDataset(inputList, targetList);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public MLP trainAndSaveMLP() {
        MLP mlp = trainMLPFromCSV();
        if (mlp != null) {
            mlpModel = mlp;
            saveModel(mlpModel, pathToMLPModel);
            return mlpModel;
        }
        else {
            return null;
        }
    }

    public float[] getInputVector(BufferedImage canvas) {
        binaryPixels = readPixelsFromCanvas(canvas);
        return convertToFloatVector(binaryPixels);
    }

    public void saveModel(MLP mlp, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(mlp);
            System.out.println("Model successfully saved!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int[][] readPixelsFromCanvas(BufferedImage canvas) {
        int cellSize = WIDTH / GRID;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                int blackCount = 0;
                for (int dy = 0; dy < cellSize; dy++) {
                    for (int dx = 0; dx < cellSize; dx++) {
                        int px = x * cellSize + dx;
                        int py = y * cellSize + dy;
                        int color = canvas.getRGB(px, py) & 0xFF;
                        if (color < 128) {
                            blackCount++;
                        }
                    }
                }
                double ratio = blackCount / (double)(cellSize * cellSize);
                binaryPixels[y][x] = (ratio > 0.2) ? 1 : 0;
            }
        }
        return binaryPixels;
    }

    public MLP loadModel(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            MLP mlp = (MLP) ois.readObject();
            System.out.println("MLP Model successfully downloaded");
            return mlp;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void printPixelsToConsole() {
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                System.out.print(binaryPixels[y][x] == 1 ? "██" : "░░");
            }
            System.out.println();
        }
    }

    public void savePixelsToCSV(String label, String csvFile) {
        try (FileWriter fw = new FileWriter(csvFile, true)) {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(",");
            for (int y = 0; y < GRID; y++) {
                for (int x = 0; x < GRID; x++) {
                    sb.append(binaryPixels[y][x]);
                    if (!(y == GRID - 1 && x == GRID - 1)) {
                        sb.append(",");
                    }
                }
            }
            sb.append("\n");
            fw.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public MLPDataset parseMLPDatasetFromCSV() {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        try {

            List<String> allLines = Files.readAllLines(Paths.get(pathToDataset));
            int totalLines = allLines.size();

            for (int i = 0; i < totalLines - 100; i++) {
                String line = allLines.get(i);
                String[] parts = line.split(",");
                if (parts.length != 1 + (GRID * GRID)) {
                    continue;
                }
                String labelStr = parts[0].trim();
                float[] inVec = new float[GRID * GRID];
                for (int j = 0; j < GRID * GRID; j++) {
                    inVec[j] = Float.parseFloat(parts[j + 1]);
                }
                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0 || classIndex >= 3) {
                    continue;
                }
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;
                inputList.add(inVec);
                targetList.add(targetVec);
            }
            return new MLPDataset(inputList, targetList);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
   /* public MLPDataset parseMLPDatasetFromCSV() {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();


        try (BufferedReader br = new BufferedReader(new FileReader(pathToDataset))) {
            String line;
            while ((line = br.readLine()) != null) {

                String[] parts = line.split(",");
                if (parts.length != 1 + (GRID * GRID)) {
                    continue;
                }
                String labelStr = parts[0].trim();
                float[] inVec = new float[GRID * GRID];
                for (int i = 0; i < GRID * GRID; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }
                int classIndex = symbolToIndex(labelStr);

                if (classIndex < 0 || classIndex >= 3) {
                    continue;
                }
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;
                inputList.add(inVec);
                targetList.add(targetVec);
            }
            return new MLPDataset(inputList, targetList);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }*/

    public MLP trainMLPFromCSV() {
        MLPDataset mlpDataset = parseMLPDatasetFromCSV();

        if (mlpDataset == null) {
            System.err.println("Failed to parse MLP Dataset from file " + pathToMLPModel);
            return null;
        }

        if (mlpDataset.inputList.isEmpty()) {
            System.err.println("Input List is empty");
            return null;
        }

        float[][] inputs = mlpDataset.inputList.toArray(new float[0][]);
        float[][] targets = mlpDataset.targetList.toArray(new float[0][]);

        MLP mlp = new MLP(GRID * GRID, 256, 3);
        mlp.train(inputs, targets, 1000, 0.001f);
        return mlp;
    }

    public int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        return switch (s) {
            case "a" -> 0;
            case "4" -> 1;
            case "f" -> 2;
            default -> -1;
        };
    }

    public String indexToSymbol(int idx) {
        return switch(idx) {
            case 0 -> "a";
            case 1 -> "4";
            case 2 -> "f";
            default -> null;
        };
    }
    public float[] convertToFloatVector(int[][] arr) {
        float[] vec = new float[GRID*GRID];
        int index = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                vec[index++] = arr[y][x];
            }
        }
        return vec;
    }
}



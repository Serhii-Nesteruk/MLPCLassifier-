package org.example;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class TrainApp {

    public static void main(String[] args) {
        String csvPath = "dataset.csv";       // Ваш CSV
        String modelPath = "mlpModel.bin";    // Куди зберегти модель

        // 1. Зчитаємо CSV
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        int GRID = 28;  // 28x28
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != 1 + (GRID * GRID)) {
                    continue; // Пропускаємо некоректні рядки
                }
                // parts[0] - label
                String labelStr = parts[0];
                float[] inputVec = new float[GRID * GRID];
                for (int i = 0; i < GRID * GRID; i++) {
                    inputVec[i] = Float.parseFloat(parts[i + 1]);
                }

                // Конвертуємо label -> index (0..35)
                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0 || classIndex > 35) {
                    // Якщо це невідомий символ
                    continue;
                }

                float[] targetVec = new float[36];
                targetVec[classIndex] = 1.0f;

                inputList.add(inputVec);
                targetList.add(targetVec);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        if (inputList.isEmpty()) {
            System.err.println("Немає даних для тренування!");
            return;
        }

        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);

        // 2. Створюємо MLP
        MLP mlp = new MLP(784, 64, 36);

        // 3. Запускаємо тренування
        mlp.train(inputs, targets, 20, 0.01f);
        // epochs=20, learningRate=0.01, це демо-параметри

        // 4. Зберігаємо модель у файл
        saveModel(modelPath, mlp);

        System.out.println("=== Тренування завершено! Модель збережено у " + modelPath);
    }

    private static int symbolToIndex(String s) {
        // Якщо цифра 0..9
        if (s.matches("\\d")) {
            return Integer.parseInt(s);
        }
        // Якщо літера A..Z (або a..z)
        s = s.toUpperCase();
        if (s.matches("[A-Z]")) {
            return 10 + (s.charAt(0) - 'A');
        }
        return -1;
    }

    // Збереження MLP у файл
    private static void saveModel(String filePath, MLP mlp) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(mlp);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

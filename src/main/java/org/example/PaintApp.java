package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class PaintApp extends JFrame {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 28; // 28x28
    private BufferedImage canvas;
    private Graphics2D g2;

    // Тут зберігатимемо пікселі, зняті з полотна
    private int[][] binaryPixels = new int[GRID][GRID];

    // Поле для введення мітки (цифра чи літера)
    private JTextField labelField;

    // Змінна для збереження моделі
    private MLP mlpModel;

    public PaintApp() {
        setTitle("Малювання + збереження у CSV + MLP");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        // Створюємо полотно
        canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, WIDTH, HEIGHT); // Білий фон
        g2.setColor(Color.BLACK);         // Колір малювання

        // Панель малювання
        DrawingPanel drawingPanel = new DrawingPanel();
        add(drawingPanel, BorderLayout.CENTER);

        // Кнопка очищення
        JButton clearButton = new JButton("Очистити");
        clearButton.addActionListener(e -> {
            g2.setColor(Color.WHITE);
            g2.fillRect(0, 0, WIDTH, HEIGHT);
            g2.setColor(Color.BLACK);
            drawingPanel.repaint();
        });

        // Кнопка для перегляду (вивід у консоль)
        JButton previewButton = new JButton("Переглянути");
        previewButton.addActionListener(e -> {
            readPixelsFromCanvas();
            printPixelsToConsole();
        });

        // Поле введення мітки
        labelField = new JTextField(5);
        labelField.setToolTipText("Введіть цифру або літеру");

        // Кнопка збереження у CSV
        JButton saveButton = new JButton("Зберегти у CSV");
        saveButton.addActionListener(e -> {
            readPixelsFromCanvas();
            String label = labelField.getText().trim();
            if (!label.isEmpty()) {
                savePixelsToCSV(label, binaryPixels, "dataset.csv");
                JOptionPane.showMessageDialog(this, "Збережено з міткою: " + label);
            } else {
                JOptionPane.showMessageDialog(this, "Будь ласка, введіть мітку (0-9 або A-Z).");
            }
        });

        // Кнопка навчання
        JButton trainButton = new JButton("Навчити модель");
        trainButton.addActionListener(e -> {
            mlpModel = trainMLPFromCSV("dataset.csv");
            if (mlpModel != null) {
                JOptionPane.showMessageDialog(this, "Навчання завершено!");
            }
        });

        // Кнопка розпізнавання
        JButton predictButton = new JButton("Розпізнати");
        predictButton.addActionListener(e -> {
            if (mlpModel == null) {
                JOptionPane.showMessageDialog(this, "Спочатку навчіть модель або завантажте існуючу!");
                return;
            }
            // Знімаємо пікселі
            readPixelsFromCanvas();
            float[] inputVector = convertToFloatVector(binaryPixels);
            int predictedIndex = mlpModel.predict(inputVector);
            // Перетворюємо індекс у символ
            String symbol = indexToSymbol(predictedIndex);
            JOptionPane.showMessageDialog(this, "Модель вважає, що це: " + symbol);
        });

        // Панель з кнопками
        JPanel controlPanel = new JPanel();
        controlPanel.add(clearButton);
        controlPanel.add(previewButton);
        controlPanel.add(new JLabel("Мітка:"));
        controlPanel.add(labelField);
        controlPanel.add(saveButton);
        controlPanel.add(trainButton);
        controlPanel.add(predictButton);

        add(controlPanel, BorderLayout.SOUTH);
        setVisible(true);
    }

    // ---------------------
    // 1. Панель малювання
    // ---------------------
    class DrawingPanel extends JPanel {
        public DrawingPanel() {
            setPreferredSize(new Dimension(WIDTH, HEIGHT));

            addMouseMotionListener(new MouseMotionAdapter() {
                public void mouseDragged(MouseEvent e) {
                    int size = 12; // "пензель"
                    g2.fillOval(e.getX() - size / 2, e.getY() - size / 2, size, size);
                    repaint();
                }
            });
        }

        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(canvas, 0, 0, null);
        }
    }

    // -----------------------------------------
    // 2. Зчитування 28x28 із canvas -> binaryPixels
    // -----------------------------------------
    private void readPixelsFromCanvas() {
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
                double ratio = blackCount / (double) (cellSize * cellSize);
                binaryPixels[y][x] = (ratio > 0.2) ? 1 : 0;
            }
        }
    }

    // ---------------------------------------------
    // 3. Вивести зображення у консоль (для перевірки)
    // ---------------------------------------------
    private void printPixelsToConsole() {
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                if (binaryPixels[y][x] == 1) {
                    System.out.print("██");
                } else {
                    System.out.print("░░");
                }
            }
            System.out.println();
        }
    }

    // -------------------------------------------
    // 4. Зберегти 28x28 у CSV
    // -------------------------------------------
    private void savePixelsToCSV(String label, int[][] pixels, String csvFilePath) {
        try (FileWriter writer = new FileWriter(csvFilePath, true)) {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(",");
            for (int y = 0; y < GRID; y++) {
                for (int x = 0; x < GRID; x++) {
                    sb.append(pixels[y][x]);
                    if (!(y == GRID - 1 && x == GRID - 1)) {
                        sb.append(",");
                    }
                }
            }
            sb.append("\n");
            writer.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // -------------------------------------------
    // 5. Тренування MLP з CSV
    // -------------------------------------------
    private MLP trainMLPFromCSV(String csvFile) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != (1 + GRID * GRID)) {
                    continue;
                }
                String labelStr = parts[0];
                float[] inputVec = new float[GRID * GRID];
                for (int i = 1; i < parts.length; i++) {
                    inputVec[i - 1] = Float.parseFloat(parts[i]);
                }

                int classIndex = symbolToIndex(labelStr);
                // Уявімо, що у нас 36 класів: 0..9, A..Z
                if (classIndex < 0 || classIndex > 35) {
                    continue;
                }
                float[] targetVec = new float[36];
                targetVec[classIndex] = 1.0f;

                inputList.add(inputVec);
                targetList.add(targetVec);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        if (inputList.isEmpty()) {
            System.err.println("Немає даних для тренування!");
            return null;
        }

        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);

        // Створюємо модель MLP
        MLP mlp = new MLP(784, 64, 36);
        mlp.train(inputs, targets, 20, 0.01f);
        return mlp;
    }

    // -------------------------------------------
    // 6. Сервісні методи для перетворення символів
    // -------------------------------------------
    private int symbolToIndex(String s) {
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

    private String indexToSymbol(int index) {
        if (index >= 0 && index <= 9) {
            return String.valueOf(index);
        } else if (index >= 10 && index < 36) {
            char c = (char) ('A' + (index - 10));
            return String.valueOf(c);
        }
        return "?";
    }

    // Перетворення 2D масиву (0 або 1) у вектор float
    private float[] convertToFloatVector(int[][] pixels) {
        float[] arr = new float[GRID * GRID];
        int idx = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                arr[idx++] = pixels[y][x];
            }
        }
        return arr;
    }
}


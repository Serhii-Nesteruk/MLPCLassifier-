package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class PredictApp extends JFrame {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 28;
    private BufferedImage canvas;
    private Graphics2D g2;
    private int[][] binaryPixels = new int[GRID][GRID];

    // Завантажена модель
    private MLP mlpModel;

    public PredictApp(MLP mlp) {
        this.mlpModel = mlp;

        setTitle("Розпізнавання символів (MLP)");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, WIDTH, HEIGHT);
        g2.setColor(Color.BLACK);

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

        // Кнопка перегляду (у консоль)
        JButton previewButton = new JButton("Переглянути");
        previewButton.addActionListener(e -> {
            readPixelsFromCanvas();
            printPixelsToConsole();
        });

        // Кнопка розпізнавання
        JButton predictButton = new JButton("Розпізнати");
        predictButton.addActionListener(e -> {
            readPixelsFromCanvas();
            float[] inputVec = convertToFloatVector(binaryPixels);
            int index = mlpModel.predict(inputVec);
            String symbol = indexToSymbol(index);
            JOptionPane.showMessageDialog(this, "MLP думає, що це: " + symbol);
        });

        JPanel controlPanel = new JPanel();
        controlPanel.add(clearButton);
        controlPanel.add(previewButton);
        controlPanel.add(predictButton);

        add(controlPanel, BorderLayout.SOUTH);
        setVisible(true);
    }

    // 1. Панель малювання
    class DrawingPanel extends JPanel {
        public DrawingPanel() {
            setPreferredSize(new Dimension(WIDTH, HEIGHT));
            addMouseMotionListener(new MouseMotionAdapter() {
                public void mouseDragged(MouseEvent e) {
                    int size = 12;
                    g2.fillOval(e.getX() - size/2, e.getY() - size/2, size, size);
                    repaint();
                }
            });
        }
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(canvas, 0, 0, null);
        }
    }

    // 2. Зчитати 28x28
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
                double ratio = blackCount / (double)(cellSize * cellSize);
                binaryPixels[y][x] = ratio > 0.2 ? 1 : 0;
            }
        }
    }

    // 3. Для перевірки у консоль
    private void printPixelsToConsole() {
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                System.out.print(binaryPixels[y][x] == 1 ? "██" : "░░");
            }
            System.out.println();
        }
    }

    // 4. Перетворити 28x28 int[][] -> float[]
    private float[] convertToFloatVector(int[][] arr) {
        float[] vector = new float[GRID * GRID];
        int idx = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                vector[idx++] = arr[y][x];
            }
        }
        return vector;
    }

    // 5. Індекс -> символ (та ж логіка, що й у TrainApp)
    private String indexToSymbol(int index) {
        if (index >= 0 && index <= 9) {
            return String.valueOf(index);
        } else if (index >= 10 && index < 36) {
            char c = (char)('A' + (index - 10));
            return String.valueOf(c);
        }
        return "?";
    }

    // ---------------------------
    // 6. Головний метод (main)
    // ---------------------------
    public static void main(String[] args) {
        // 1. Завантажуємо модель
        MLP model = loadModel("mlpModel.bin"); // Шлях до файлу моделі
        if (model == null) {
            System.err.println("Помилка: не вдалося завантажити модель!");
            return;
        }
        // 2. Запускаємо GUI
        SwingUtilities.invokeLater(() -> new PredictApp(model));
    }

    // Завантаження MLP з файлу
    private static MLP loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (MLP) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}

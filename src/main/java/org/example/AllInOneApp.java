package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class AllInOneApp extends JFrame {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 56;
    private BufferedImage canvas;
    private Graphics2D g2;
    private int[][] binaryPixels = new int[GRID][GRID];

    // Поле введення мітки
    private JTextField labelField;

    // Змінна для MLP-моделі (якщо/коли натиснемо «Навчити»)
    private MLP mlpModel;

    public AllInOneApp() {
        setTitle("Усе-в-одному: Малювання, CSV, Тренування, Розпізнавання (2 ряди кнопок)");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        // 1. Ініціалізація полотна
        canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, WIDTH, HEIGHT);
        g2.setColor(Color.BLACK);

        // 2. Панель малювання
        DrawingPanel drawingPanel = new DrawingPanel();
        add(drawingPanel, BorderLayout.CENTER);

        String pathToModel = "mlpModel.bin";
        File file = new File(pathToModel);
        if (file.exists()) {
            mlpModel = loadModel(pathToModel);
        }
        else {
            System.out.println("File not exists");
        }

        // -------------------------
        // 3. Створюємо панель для кнопок (BoxLayout по Y)
        // -------------------------
        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new BoxLayout(buttonPanel, BoxLayout.Y_AXIS));

        // (a) Створюємо перший "ряд" кнопок
        JPanel rowPanel1 = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 5));
        // (b) Створюємо другий "ряд" кнопок
        JPanel rowPanel2 = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 5));

        // 4. Створюємо кнопки
        JButton clearBtn = new JButton("Очистити");
        clearBtn.addActionListener(e -> {
            g2.setColor(Color.WHITE);
            g2.fillRect(0, 0, WIDTH, HEIGHT);
            g2.setColor(Color.BLACK);
            drawingPanel.repaint();
        });

        JButton previewBtn = new JButton("Переглянути");
        previewBtn.addActionListener(e -> {
            readPixelsFromCanvas();
            printPixelsToConsole();
        });

        JButton predictBtn = new JButton("Розпізнати");
        predictBtn.addActionListener(e -> {

            String _pathToModel = "mlpModel.bin";
            File _file = new File(pathToModel);
            if (file.exists()) {
                mlpModel = loadModel(pathToModel);
            }
            else {
                System.out.println("File not exists");
            }
            readPixelsFromCanvas();
            float[] inputVec = convertToFloatVector(binaryPixels);
            PredictionResult result = mlpModel.predict(inputVec);
            String symbol = indexToSymbol(result.predictedIndex);

            if (result.confidence < 0.5) {
                JOptionPane.showMessageDialog(this, "MLP не може розпізнати цей символ: ");
            }
            else {
                JOptionPane.showMessageDialog(this, "MLP думає, що це: " + symbol + " З ймовірністю " + result.confidence);
            }
        });

        labelField = new JTextField(3);
        labelField.setToolTipText("Введіть цифру (0..9) чи літеру (A..Z)");

        JButton saveBtn = new JButton("Зберегти у CSV");
        saveBtn.addActionListener(e -> {
            readPixelsFromCanvas();
            String label = labelField.getText().trim();
            if (!label.isEmpty()) {
                savePixelsToCSV(label, "dataset.csv");
                JOptionPane.showMessageDialog(this, "Збережено з міткою [" + label + "]");
            } else {
                JOptionPane.showMessageDialog(this, "Будь ласка, введіть мітку!");
            }
        });

        JButton trainBtn = new JButton("Навчити MLP");
        trainBtn.addActionListener(e -> {
            mlpModel = trainMLPFromCSV("dataset.csv");
            if (mlpModel != null) {
                JOptionPane.showMessageDialog(this, "MLP навчено!");
                saveModel(mlpModel, "mlpModel.bin");
            }
        });

        // 5. Розподіляємо кнопки у два "ряди"
        // Приклад: у перший ряд винесемо три кнопки
        rowPanel1.add(clearBtn);
        rowPanel1.add(previewBtn);
        rowPanel1.add(predictBtn);

        // У другий ряд - мітка, поле, "Зберегти", "Навчити"
        rowPanel2.add(new JLabel("Мітка:"));
        rowPanel2.add(labelField);
        rowPanel2.add(saveBtn);
        rowPanel2.add(trainBtn);

        // 6. Додаємо обидва рядки до buttonPanel
        buttonPanel.add(rowPanel1);
        buttonPanel.add(rowPanel2);

        // 7. Додаємо buttonPanel в низ (SOUTH)
        add(buttonPanel, BorderLayout.SOUTH);

        setVisible(true);
    }

    // ---------------------------
    // Панель малювання
    // ---------------------------
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

    private void saveModel(MLP mlp, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(mlp);
            System.out.println("Модель успішно збережена!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

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
                binaryPixels[y][x] = (ratio > 0.2) ? 1 : 0;
            }
        }
    }

    private MLP loadModel(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            MLP mlp = (MLP) ois.readObject();
            System.out.println("Модель успішно завантажена!");
            return mlp;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    // 2. Показати 28x28 у консолі
    private void printPixelsToConsole() {
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                System.out.print(binaryPixels[y][x] == 1 ? "██" : "░░");
            }
            System.out.println();
        }
    }

    // 3. Зберегти у CSV
    private void savePixelsToCSV(String label, String csvFile) {
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

    // 4. Навчити MLP на основі CSV
// 4. Навчити MLP на основі CSV (3 класи)
    private MLP trainMLPFromCSV(String csvFile) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Формат: label, p0, p1, ... p3135 (56x56)
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
                // Якщо classIndex не належить [0,2], пропускаємо цей приклад
                if (classIndex < 0 || classIndex >= 3) {
                    continue;
                }
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;
                inputList.add(inVec);
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

        // Створюємо та тренуємо MLP (3136->256->3)
        MLP mlp = new MLP(GRID * GRID, 256, 3);
        mlp.train(inputs, targets, 600, 0.0001f);
        return mlp;
    }

    // 5. Допоміжні методи
    // (symbol -> index, index -> symbol, перетворення пікселів у float[])
    private int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        if (s.equals("a")) {
            return 0;
        } else if (s.equals("4")) {
            return 1;
        } else if (s.equals("f")) {
            return 2;
        }
        return -1; // Недопустима мітка
    }

    private String indexToSymbol(int idx) {
        if (idx == 0) return "a";
        if (idx == 1) return "4";
        if (idx == 2) return "f";
        return "?";
    }
    private float[] convertToFloatVector(int[][] arr) {
        float[] vec = new float[GRID*GRID];
        int index = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                vec[index++] = arr[y][x];
            }
        }
        return vec;
    }

    // ---------------------------
    // Точка входу
    // ---------------------------
    public static void main(String[] args) {
        SwingUtilities.invokeLater(AllInOneApp::new);
    }
}



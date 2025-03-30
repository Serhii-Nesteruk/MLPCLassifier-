package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MLPClassifier extends JFrame {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 56;
    private BufferedImage canvas;
    private Graphics2D g2;
    private int[][] binaryPixels = new int[GRID][GRID];
    private DrawingPanel drawingPanel;
    private final String pathToMLPModel = "mlpModel.bin";
    private JButton clearBtn, previewBtn, predictBtn, saveBtn, trainBtn;
    private JPanel rowPanel1, rowPanel2, buttonPanel;
    private JTextField labelField;

    private MLP mlpModel;

    public MLPClassifier() {
        setTitle("Усе-в-одному: Малювання, CSV, Тренування, Розпізнавання (2 ряди кнопок)");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        labelField = new JTextField(3);
        labelField.setToolTipText("Введіть цифру (0..9) чи літеру (A..Z)");

        initComponents();

        setVisible(true);
    }

    private void initComponents() {
        loadMLPModel();
        initButtons();
        initPanels();
    }

    private void initPanels() {
        initDrawingPanel();
        initRowPanel();
        initButtonPanel();
    }

    private void loadMLPModel() {
        File file = new File(pathToMLPModel);
        if (file.exists()) {
            mlpModel = loadModel(pathToMLPModel);
        }
        else {
            System.err.println("File not exists");
        }

    }

    private void initRowPanel() {
        rowPanel1 = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 5));
        rowPanel2 = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 5));

        rowPanel1.add(clearBtn);
        rowPanel1.add(previewBtn);
        rowPanel1.add(predictBtn);

        rowPanel2.add(new JLabel("Мітка:"));
        rowPanel2.add(labelField);
        rowPanel2.add(saveBtn);
        rowPanel2.add(trainBtn);
    }

    private void initButtonPanel() {
        buttonPanel = new JPanel();
        buttonPanel.setLayout(new BoxLayout(buttonPanel, BoxLayout.Y_AXIS));

        buttonPanel.add(rowPanel1);
        buttonPanel.add(rowPanel2);

        add(buttonPanel, BorderLayout.SOUTH);

    }

    private void initButtons() {
        clearBtn = new JButton("Очистити");
        clearBtn.addActionListener(e -> {
            g2.setColor(Color.WHITE);
            g2.fillRect(0, 0, WIDTH, HEIGHT);
            g2.setColor(Color.BLACK);
            drawingPanel.repaint();
        });

        previewBtn = new JButton("Переглянути");
        previewBtn.addActionListener(e -> {
            readPixelsFromCanvas();
            printPixelsToConsole();
        });

        predictBtn = new JButton("Розпізнати");
        predictBtn.addActionListener(e -> {
            File _file = new File(pathToMLPModel);
            if (_file.exists() && mlpModel == null) {
                mlpModel = loadModel(pathToMLPModel);
            }
            else {
                System.out.println("File not exists");
            }
            readPixelsFromCanvas();
            float[] inputVec = convertToFloatVector(binaryPixels);
            PredictionResult result = mlpModel.predict(inputVec);
            String symbol = indexToSymbol(result.predictedIndex);

            if (symbol == null) {
                System.err.println("Failed to parse index to symbol");

            }

            if (result.confidence < 0.5) {
                JOptionPane.showMessageDialog(this, "MLP не може розпізнати цей символ: ");
            }
            else {
                JOptionPane.showMessageDialog(this, "MLP думає, що це: " + symbol + " З ймовірністю " + result.confidence);
            }
        });
        saveBtn = new JButton("Зберегти у CSV");
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

        trainBtn = new JButton("Навчити MLP");
        trainBtn.addActionListener(e -> {
            mlpModel = trainMLPFromCSV("dataset.csv");
            if (mlpModel != null) {
                JOptionPane.showMessageDialog(this, "MLP навчено!");
                saveModel(mlpModel, "mlpModel.bin");
            }
        });
    }

    private void initDrawingPanel() {
        canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, WIDTH, HEIGHT);
        g2.setColor(Color.BLACK);

        drawingPanel = new DrawingPanel();
        add(drawingPanel, BorderLayout.CENTER);
    }

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

    public MLPDataset parseMLPDatasetFromCSV(String csvFile) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();


        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
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
    }

    private MLP trainMLPFromCSV(String csvFile) {
        MLPDataset mlpDataset = parseMLPDatasetFromCSV(csvFile);

        if (mlpDataset == null) {
            System.err.println("Failed to parse MLP Dataset from file " + csvFile);
            return null;
        }

        if (mlpDataset.inputList.isEmpty()) {
            System.err.println("Input List is empty");
            return null;
        }

        float[][] inputs = mlpDataset.inputList.toArray(new float[0][]);
        float[][] targets = mlpDataset.targetList.toArray(new float[0][]);

        MLP mlp = new MLP(GRID * GRID, 256, 3);
        mlp.train(inputs, targets, 600, 0.0001f);
        return mlp;
    }

    private int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        return switch (s) {
            case "a" -> 0;
            case "4" -> 1;
            case "f" -> 2;
            default -> -1;
        };
    }

    private String indexToSymbol(int idx) {
        return switch(idx) {
            case 0 -> "a";
            case 1 -> "4";
            case 2 -> "f";
            default -> null;
        };
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
        SwingUtilities.invokeLater(MLPClassifier::new);
    }
}



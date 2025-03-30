package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;

public class UI extends JFrame {

    private final int WIDTH = 500, HEIGHT = 500;
    private final int GRID = 56;
    private BufferedImage canvas;
    private Graphics2D g2;
    private JButton clearBtn, previewBtn, predictBtn, saveBtn, trainBtn;
    private JPanel rowPanel1, rowPanel2, buttonPanel;
    private JTextField labelField;
    private DrawingPanel drawingPanel;
    private int[][] binaryPixels = new int[GRID][GRID];

    private MLPClassifier mlpClassifier;

    private MLP mlpModel;

    public UI() {
        mlpClassifier = new MLPClassifier();

        setTitle("MLPClassifier");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        mlpModel = mlpClassifier.loadModel(mlpClassifier.pathToMLPModel);

        labelField = new JTextField(3);
        labelField.setToolTipText("Введіть цифру (0..9) чи літеру (A..Z)");

        initComponents();

        setVisible(true);
    }

    public void initComponents() {
        initButtons();
        initPanels();
    }
    private void initPanels() {
        initDrawingPanel();
        initRowPanel();
        initButtonPanel();
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
            binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
            mlpClassifier.printPixelsToConsole();
        });

        predictBtn = new JButton("Розпізнати");
        predictBtn.addActionListener(e -> {
            File _file = new File(mlpClassifier.pathToMLPModel);
            if (_file.exists() && mlpModel == null) {
                mlpModel = mlpClassifier.loadModel(mlpClassifier.pathToMLPModel);
            }
            else {
                System.out.println("File not exists");
            }
            binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
            float[] inputVec = mlpClassifier.convertToFloatVector(binaryPixels);
            PredictionResult result = mlpModel.predict(inputVec);
            String symbol = mlpClassifier.indexToSymbol(result.predictedIndex);

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
            binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
            String label = labelField.getText().trim();
            if (!label.isEmpty()) {
                mlpClassifier.savePixelsToCSV(label, "dataset.csv");
                JOptionPane.showMessageDialog(this, "Збережено з міткою [" + label + "]");
            } else {
                JOptionPane.showMessageDialog(this, "Будь ласка, введіть мітку!");
            }
        });

        trainBtn = new JButton("Навчити MLP");
        trainBtn.addActionListener(e -> {
            mlpModel = mlpClassifier.trainMLPFromCSV("dataset.csv");
            if (mlpModel != null) {
                JOptionPane.showMessageDialog(this, "MLP навчено!");
                mlpClassifier.saveModel(mlpModel, "mlpModel.bin");
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

}

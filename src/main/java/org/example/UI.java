package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;

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

    private void clearBtnActionListener() {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, WIDTH, HEIGHT);
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    private void previewBtnActionListener() {
        binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
        mlpClassifier.printPixelsToConsole();
    }

    private void predictBtnActionListener() {
        float[] inputVec = mlpClassifier.getInputVector(canvas);
        PredictionResult result = mlpClassifier.predict(inputVec);
        String symbol = mlpClassifier.indexToSymbol(result.predictedIndex);

        if (symbol == null) {
            System.err.println("Failed to parse index to symbol");
        }

        if (result.confidence < 0.5) {
            JOptionPane.showMessageDialog(this, "MLP can't classify this symbol: ");
        }
        else {
            JOptionPane.showMessageDialog(this, "MLP думає, що це: " + symbol + " З ймовірністю " + result.confidence);
        }
    }

    private void saveBtnActionListener() {
        binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
        String label = labelField.getText().trim();
        if (!label.isEmpty()) {
            mlpClassifier.savePixelsToCSV(label, "dataset.csv");
            JOptionPane.showMessageDialog(this, "Збережено з міткою [" + label + "]");
        } else {
            JOptionPane.showMessageDialog(this, "Будь ласка, введіть мітку!");
        }
    }

    private void trainBtnActionListener() {
        if (mlpClassifier.trainAndSaveMLP() != null)
            JOptionPane.showMessageDialog(this, "MLP successfully trained");
    }

    private void initButtons() {
        clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> clearBtnActionListener());

        previewBtn = new JButton("Show in Console");
        previewBtn.addActionListener(e -> previewBtnActionListener());

        predictBtn = new JButton("Predict");
        predictBtn.addActionListener(e -> predictBtnActionListener());

        saveBtn = new JButton("Save");
        saveBtn.addActionListener(e -> saveBtnActionListener());

        trainBtn = new JButton("Train MLP");
        trainBtn.addActionListener(e -> trainBtnActionListener());
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

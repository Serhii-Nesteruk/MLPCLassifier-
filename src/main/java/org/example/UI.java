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
    private JButton clearBtn, previewBtn, predictBtn, saveBtn, trainBtn, testBtn;
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
        labelField.setToolTipText("Enter a number (0..9) or a letter (A..Z)");

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

        rowPanel2.add(new JLabel("Tag:"));
        rowPanel2.add(labelField);
        rowPanel2.add(saveBtn);
        rowPanel2.add(trainBtn);
        rowPanel2.add(testBtn);
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

        if (result.confidence < 0.75) {
            JOptionPane.showMessageDialog(this, "MLP can't classify this symbol: ");
        }
        else {
            JOptionPane.showMessageDialog(this, "MLP thinks it is: " + symbol + " With probability" + result.confidence);
        }
    }

    private void saveBtnActionListener() {
        binaryPixels = mlpClassifier.readPixelsFromCanvas(canvas);
        String label = labelField.getText().trim();
        if (!label.isEmpty()) {
            mlpClassifier.savePixelsToCSV(label, "dataset.csv");
            JOptionPane.showMessageDialog(this, "Saved with tag [" + label + "]");
        } else {
            JOptionPane.showMessageDialog(this, "Please enter a tag!");
        }
    }

    private void trainBtnActionListener() {
        if (mlpClassifier.trainAndSaveMLP() != null)
            JOptionPane.showMessageDialog(this, "MLP successfully trained");
    }

    public void testBtnActionListener() {
        int countOfGoodAnswers = 0;

        MLPDataset mlpDataset = mlpClassifier.loadTestSample();
        float[][] inputs = mlpDataset.inputList.toArray(new float[0][]);
        float[][] targets = mlpDataset.targetList.toArray(new float[0][]);

        for (int i = 0; i < inputs.length; i++) {
            PredictionResult result = mlpClassifier.predict(inputs[i]);
            int predictedIndex = result.predictedIndex;

            int trueIndex = -1;
            for (int j = 0; j < targets[i].length; j++) {
                if (targets[i][j] == 1.0f) {
                    trueIndex = j;
                    break;
                }
            }

            if (predictedIndex == trueIndex) {
                countOfGoodAnswers++;
            }
        }
        System.out.println("Number of correct answers: " + countOfGoodAnswers + "/" + inputs.length);
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

        testBtn = new JButton("Test");
        testBtn.addActionListener(e -> testBtnActionListener());
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

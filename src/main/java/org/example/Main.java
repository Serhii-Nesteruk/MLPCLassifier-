package org.example;

import javax.swing.SwingUtilities;

public class Main {
    public static void main(String[] args) {
        // Запускаємо додаток із графічним інтерфейсом
        SwingUtilities.invokeLater(PaintApp::new);
    }
}

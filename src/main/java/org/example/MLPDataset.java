package org.example;

import java.util.ArrayList;
import java.util.List;

public class MLPDataset {
    public List<float[]> inputList;
    public List<float[]> targetList;

    public MLPDataset(List<float[]> inputList, List<float[]> targetList) {
        this.inputList = inputList;
        this.targetList = targetList;
    }
}

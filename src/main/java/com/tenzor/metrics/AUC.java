package com.tenzor.metrics;

import com.tenzor.core.Tensor;
import java.util.*;

public class AUC {
    public static double compute(Tensor yTrue, Tensor yScore) {
        int n = yTrue.size();
        double[][] pairs = new double[n][2];
        for (int i = 0; i < n; i++) {
            pairs[i][0] = yScore.get(i);
            pairs[i][1] = yTrue.get(i);
        }
        Arrays.sort(pairs, (a, b) -> Double.compare(b[0], a[0]));

        double pos = 0, neg = 0;
        for (int i = 0; i < n; i++) {
            if (pairs[i][1] == 1.0) pos++;
            else neg++;
        }

        if (pos == 0 || neg == 0) return 0.5;

        double rankSum = 0;
        for (int i = 0; i < n; i++) {
            if (pairs[i][1] == 1.0) rankSum += (i + 1);
        }

        return (rankSum - pos * (pos + 1) / 2) / (pos * neg);
    }
}

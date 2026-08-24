package com.tenzor.data;

import com.tenzor.core.Tensor;

public interface Dataset {
    int size();
    Tensor getX(int index);
    Tensor getY(int index);
}

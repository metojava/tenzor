package com.tenzor.io;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.google.gson.*;
import java.io.*;
import java.nio.file.*;

public class TensorSaver {
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public static void save(Tensor tensor, String filePath) throws IOException {
        JsonObject obj = new JsonObject();
        JsonArray shapeArr = new JsonArray();
        for (int d : tensor.shape().dims()) shapeArr.add(d);
        obj.add("shape", shapeArr);

        JsonArray dataArr = new JsonArray();
        for (double v : tensor.getData()) dataArr.add(v);
        obj.add("data", dataArr);

        try (Writer w = new FileWriter(filePath)) {
            gson.toJson(obj, w);
        }
    }

    public static Tensor load(String filePath) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(filePath)));
        JsonObject obj = gson.fromJson(content, JsonObject.class);

        JsonArray sArr = obj.getAsJsonArray("shape");
        int[] dims = new int[sArr.size()];
        for (int i = 0; i < sArr.size(); i++) dims[i] = sArr.get(i).getAsInt();

        JsonArray dArr = obj.getAsJsonArray("data");
        double[] data = new double[dArr.size()];
        for (int i = 0; i < dArr.size(); i++) data[i] = dArr.get(i).getAsDouble();

        return new Tensor(data, new Shape(dims));
    }

    public static void saveBinary(Tensor tensor, String filePath) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath))) {
            int[] dims = tensor.shape().dims();
            dos.writeInt(dims.length);
            for (int d : dims) dos.writeInt(d);
            for (double v : tensor.getData()) dos.writeDouble(v);
        }
    }

    public static Tensor loadBinary(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int numDims = dis.readInt();
            int[] dims = new int[numDims];
            for (int i = 0; i < numDims; i++) dims[i] = dis.readInt();
            Shape shape = new Shape(dims);
            double[] data = new double[shape.totalSize()];
            for (int i = 0; i < data.length; i++) data[i] = dis.readDouble();
            return new Tensor(data, shape);
        }
    }
}

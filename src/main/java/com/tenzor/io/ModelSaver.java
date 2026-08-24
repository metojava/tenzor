package com.tenzor.io;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.google.gson.*;
import java.io.*;
import java.util.*;

public class ModelSaver {
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public static void save(Sequential model, String filePath) throws IOException {
        JsonObject root = new JsonObject();
        JsonArray layersArr = new JsonArray();
        for (Layer layer : model.getLayers()) {
            JsonObject layerObj = new JsonObject();
            layerObj.addProperty("type", layer.getClass().getSimpleName());
            layerObj.addProperty("name", layer.getName());

            if (layer instanceof Dense) {
                Dense d = (Dense) layer;
                layerObj.addProperty("units", d.getUnits());
                layerObj.addProperty("activation", d.getActivation());
                JsonArray wData = new JsonArray();
                for (double v : d.getWeights().getData()) wData.add(v);
                layerObj.add("weights", wData);
                layerObj.add("weights_shape", toJsonArray(d.getWeights().shape().dims()));
                if (d.getBias() != null) {
                    JsonArray bData = new JsonArray();
                    for (double v : d.getBias().getData()) bData.add(v);
                    layerObj.add("bias", bData);
                }
            }

            layersArr.add(layerObj);
        }
        root.add("layers", layersArr);
        root.add("history", toJsonMap(model.getHistory()));

        try (Writer w = new FileWriter(filePath)) {
            gson.toJson(root, w);
        }
    }

    private static JsonElement toJsonArray(int[] arr) {
        JsonArray ja = new JsonArray();
        for (int v : arr) ja.add(v);
        return ja;
    }

    private static JsonElement toJsonMap(Map<String, List<Double>> map) {
        JsonObject obj = new JsonObject();
        for (Map.Entry<String, List<Double>> e : map.entrySet()) {
            JsonArray arr = new JsonArray();
            for (double v : e.getValue()) arr.add(v);
            obj.add(e.getKey(), arr);
        }
        return obj;
    }
}

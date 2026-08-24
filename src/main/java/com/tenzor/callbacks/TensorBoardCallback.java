package com.tenzor.callbacks;

import com.tenzor.nn.Sequential;
import java.io.*;
import java.nio.file.*;

public class TensorBoardCallback extends Callback {
    private String logDir;
    private boolean writeGraph;
    private Writer writer;

    public TensorBoardCallback(String logDir) {
        this(logDir, true);
    }

    public TensorBoardCallback(String logDir, boolean writeGraph) {
        this.logDir = logDir;
        this.writeGraph = writeGraph;
    }

    @Override
    public void onTrainBegin(Sequential model) {
        try {
            Path dir = Paths.get(logDir);
            Files.createDirectories(dir);
            writer = new FileWriter(dir.resolve("events.out.tfevents").toFile(), true);

            if (writeGraph) {
                StringBuilder sb = new StringBuilder();
                sb.append("graph {\n");
                for (int i = 0; i < model.getLayers().size(); i++) {
                    sb.append("  node [name=\"").append(model.getLayer(i).getName())
                      .append("\", op=\"").append(model.getLayer(i).getClass().getSimpleName())
                      .append("\"]\n");
                }
                sb.append("}\n");
                writer.write(sb.toString());
                writer.flush();
            }
        } catch (IOException e) {
            System.err.println("TensorBoard: Could not create log dir: " + e.getMessage());
        }
    }

    @Override
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {
        if (writer != null) {
            try {
                writer.write(String.format("epoch %d: loss=%.6f, val_loss=%.6f%n", epoch, loss, valLoss));
                writer.flush();
            } catch (IOException e) {
                System.err.println("TensorBoard: Write error: " + e.getMessage());
            }
        }
    }

    @Override
    public void onTrainEnd(Sequential model) {
        if (writer != null) {
            try {
                writer.close();
            } catch (IOException e) {
                // ignore
            }
        }
    }
}

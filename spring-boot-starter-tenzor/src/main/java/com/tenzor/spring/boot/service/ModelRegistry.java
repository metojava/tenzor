package com.tenzor.spring.boot.service;

import com.tenzor.nn.Sequential;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ModelRegistry {

    private final Map<String, Sequential> models = new ConcurrentHashMap<>();
    private final Map<String, ModelMetadata> metadata = new ConcurrentHashMap<>();

    public void register(String name, Sequential model) {
        register(name, model, new ModelMetadata());
    }

    public void register(String name, Sequential model, ModelMetadata meta) {
        models.put(name, model);
        metadata.put(name, meta);
    }

    public Sequential get(String name) {
        return models.get(name);
    }

    public ModelMetadata getMetadata(String name) {
        return metadata.get(name);
    }

    public boolean exists(String name) {
        return models.containsKey(name);
    }

    public void remove(String name) {
        models.remove(name);
        metadata.remove(name);
    }

    public void clear() {
        models.clear();
        metadata.clear();
    }

    public Map<String, Sequential> getAllModels() {
        return Map.copyOf(models);
    }

    public static class ModelMetadata {
        private String description = "";
        private String version = "1.0.0";
        private String inputShape = "";
        private String outputShape = "";
        private String createdAt;
        private long trainTimeMs;
        private double accuracy;

        public ModelMetadata() {
            this.createdAt = java.time.Instant.now().toString();
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getVersion() {
            return version;
        }

        public void setVersion(String version) {
            this.version = version;
        }

        public String getInputShape() {
            return inputShape;
        }

        public void setInputShape(String inputShape) {
            this.inputShape = inputShape;
        }

        public String getOutputShape() {
            return outputShape;
        }

        public void setOutputShape(String outputShape) {
            this.outputShape = outputShape;
        }

        public String getCreatedAt() {
            return createdAt;
        }

        public void setCreatedAt(String createdAt) {
            this.createdAt = createdAt;
        }

        public long getTrainTimeMs() {
            return trainTimeMs;
        }

        public void setTrainTimeMs(long trainTimeMs) {
            this.trainTimeMs = trainTimeMs;
        }

        public double getAccuracy() {
            return accuracy;
        }

        public void setAccuracy(double accuracy) {
            this.accuracy = accuracy;
        }
    }
}

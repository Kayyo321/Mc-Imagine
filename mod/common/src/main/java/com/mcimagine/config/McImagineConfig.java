package com.mcimagine.config;

public class McImagineConfig {
    public String modelDirectoryPath = "models/mcimagine";
    public String gpuPreference = "AUTO"; // AUTO, FORCE_CPU, FORCE_GPU
    public int cacheSize = 1024;
    public int maxGenerationThreads = 4;
    public String logLevel = "INFO";
}\n
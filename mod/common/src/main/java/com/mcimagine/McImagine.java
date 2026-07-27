package com.mcimagine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McImagine {
    public static final String MOD_ID = "mcimagine";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    public static void init() {
        LOGGER.info("Initializing Mc-Imagine: AI-powered world generation");
    }
}\n
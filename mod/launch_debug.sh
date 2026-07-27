#!/bin/bash
# Launch the Fabric development client in JVM Debug mode for Hot-Swapping
echo "Starting Mc-Imagine Fabric Client (Debug Mode)..."
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ./gradlew :fabric:runClient --debug-jvm

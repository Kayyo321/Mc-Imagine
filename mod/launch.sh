#!/bin/bash
# Launch the Fabric development client normally
echo "Starting Mc-Imagine Fabric Client..."
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ./gradlew :fabric:runClient

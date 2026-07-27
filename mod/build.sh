#!/bin/bash
set -e

echo "Ensuring Java 17 is installed..."
if ! brew ls --versions openjdk@17 > /dev/null; then
    echo "Installing openjdk@17 via Homebrew..."
    brew install openjdk@17
fi

export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

echo "Using Java version:"
java -version

echo "Building mod..."
./gradlew build

// Include Particle Device OS APIs
#include "Particle.h"

// Use AUTOMATIC mode to connect to cloud automatically
SYSTEM_MODE(AUTOMATIC);

// Optional: Enable logging over USB
SerialLogHandler logHandler(LOG_LEVEL_INFO);

// Define output pins for different colors (use real GPIO pins you wired)
const int RED_PIN = D2;         
const int GREEN_PIN = D3;
const int YELLOW_PIN = D5;

// Forward declaration of function
int setColor(String color);

void setup() {
  // Setup pins as outputs
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);

  // Turn both off at start
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);

  // Register the cloud function
  Particle.function("setColor", setColor);

  Log.info("Setup complete. Waiting for color commands.");
}

void loop() {
  // Nothing to do here, it's all cloud-triggered.
}

// Cloud function to receive color commands
int setColor(String color) {
  Log.info("Received color command: %s", color.c_str());

  if (color == "red") {
    digitalWrite(RED_PIN, HIGH);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(YELLOW_PIN, LOW);
  }
  else if (color == "green") {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, HIGH);
    digitalWrite(YELLOW_PIN, LOW);
  }
  else if (color == "yellow") {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(YELLOW_PIN, HIGH);
  }
  else if (color == "off") {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(YELLOW_PIN, LOW);
  }
  else {
    Log.warn("Unknown color: %s", color.c_str());
    return -1; // error
  }

  return 1; // success
}


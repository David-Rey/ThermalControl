#include <Arduino.h>
#include <Adafruit_INA260.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "arduino_freertos.h"
#include <queue.h>
#include <semphr.h>         
#include <iostream>
#include <string>

// Update Rates
#define INNER_LOOP_UPDATE_RATE 20        // 20ms = 50Hz
#define OUTTER_LOOP_UPDATE_RATE 100    // 100ms = 10Hz
#define TEMPRATURE_UPDATE_RATE 100
#define MUTEX_TIMEOUT 10

// Debug
#define INNER_LOOP_DEBUG 1
#define TEMP_LOOP_DEBUG 0
#define TEMP_READ_DEBUG 1

// Pins
#define PELTIER_IN1 5
#define PELTIER_IN2 4
#define PELTIER_EN 6

// Max Current
//const float MAX_PELTIER_CURRENT_AMPS = 3.0;

QueueHandle_t currentQueue;

// Setup current
Adafruit_INA260 ina260 = Adafruit_INA260();

struct TempData {
    float temp1;
    float temp2;
};
TempData latestTemps;

// Setup mutex
SemaphoreHandle_t xI2cMutex = xSemaphoreCreateMutex();
SemaphoreHandle_t serialMutex = xSemaphoreCreateMutex();
SemaphoreHandle_t tempMutex = xSemaphoreCreateMutex();

OneWire ds18x20[] = {14, 15};
const int oneWireCount = sizeof(ds18x20) / sizeof(OneWire);
DallasTemperature sensor[oneWireCount];

class PID{
  public:
    double Kp;
    double Ki;
    double Kd;
    double fc;
    double dt;
    double tau;
    double alpha;

    double integrator;
    double prevFilteredError;

    double clampMax;
    double clampMin;

    PID(double p, double i, double d, double fc, double dt):
      Kp(p), Ki(i), Kd(d), fc(fc), dt(dt), integrator(0.0), prevFilteredError(0.0) {
        tau = 1.0 / (2.0 * PI * fc);
        alpha = dt / (dt + tau);

        clampMax = 1e6;
        clampMin = -1e6;
      }

    void reset(){
      integrator = 0.0;
      prevFilteredError = 0.0;
    }

    void setClamps(double min, double max){
      clampMax = max;
      clampMin = min;
    }

    double update(double setPoint, double currentValue){
      // Get current error
      double error = setPoint - currentValue;
      
      // Filter Error
      double filteredError = (alpha * error) + (1.0 - alpha) * prevFilteredError;

      // Get proportional signal 
      double P = Kp * error;

      // Get integrator error
      integrator += error * dt;

      // Clamp integrator error
      if (integrator > clampMax) (integrator = clampMax);
      if (integrator < clampMin) (integrator = clampMin);

      // Get integrator signal 
      double I = Ki * integrator;

      // Filter Derivative
      double D = Kd * (filteredError - prevFilteredError) / dt;
      
      // Get final control by summing the P, I and D terms
      double control = P + I + D;
      
      // Save previous values
      prevFilteredError = filteredError;

      // Return final control
      return control;
    }
};

void setDirection(int direction){
  if (direction > 0){
    digitalWrite(PELTIER_IN1, arduino::LOW);
    digitalWrite(PELTIER_IN2, arduino::HIGH);
  }
  else{
    digitalWrite(PELTIER_IN1, arduino::HIGH);
    digitalWrite(PELTIER_IN2, arduino::LOW);
  }
}

void setPwmControl(int pwm){
  analogWrite(PELTIER_EN, pwm);
}

PID* innerPID;
PID* tempPID;

static void innerLoop(void*){
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(INNER_LOOP_UPDATE_RATE);
  float current = 0.0;
  double currentSetPoint = 0.0;
  double PWMcontrol = 0.0;

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    xQueuePeek(currentQueue, &currentSetPoint, 0);
    
    // read current
    if (xSemaphoreTake(xI2cMutex, MUTEX_TIMEOUT) == pdTRUE){
      current = ina260.readCurrent();
      xSemaphoreGive(xI2cMutex);
    }
    PWMcontrol = innerPID->update(currentSetPoint, current);

    // print
    if (INNER_LOOP_DEBUG){
      if (xSemaphoreTake(serialMutex, MUTEX_TIMEOUT) == pdTRUE){
        Serial.print("I,");
        Serial.print(currentSetPoint);
        Serial.print(",");
        Serial.print(current);
        Serial.print(",");
        Serial.println(PWMcontrol);

        xSemaphoreGive(serialMutex);
      }
    }
    setPwmControl((int)PWMcontrol);
    //vTaskDelay(pdMS_TO_TICKS(1)); // Yield to other tasks
  }
}

static void outterLoop(void*){
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(OUTTER_LOOP_UPDATE_RATE);
  
  float currentTemp = 0.0;
  double tempSetPoint = 40.0;
  double currentSet = 0.0;  // mA
  double startTime = 0.0;
  setDirection(1);

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    startTime = millis() / 1000.0;
    if (startTime > 10 && startTime < 30){
      currentSet = 1000.0;
    } else{
      currentSet = 0.0;
    }
    
    if (xSemaphoreTake(tempMutex, MUTEX_TIMEOUT) == pdTRUE){
      currentTemp = latestTemps.temp1;
      xSemaphoreGive(tempMutex);
    }

    //currentSet = tempPID->update(tempSetPoint, currentTemp);

    if (TEMP_LOOP_DEBUG){
      if (xSemaphoreTake(serialMutex, MUTEX_TIMEOUT) == pdTRUE){
        Serial.print("O,");
        Serial.print(tempSetPoint);
        Serial.print(",");
        Serial.print(currentSet);
        Serial.print(",");
        Serial.println(currentTemp);
        xSemaphoreGive(serialMutex);
      }
    }
    xQueueOverwrite(currentQueue, &currentSet);
    //vTaskDelay(pdMS_TO_TICKS(1)); // Yield to other tasks
  }
}

static void readTemp(void*){
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(TEMPRATURE_UPDATE_RATE);

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    for(int i = 0; i < oneWireCount; i++) {
      sensor[i].requestTemperatures(); 
    }
    
    if (xSemaphoreTake(tempMutex, MUTEX_TIMEOUT) == pdTRUE){
      latestTemps.temp1 = sensor[0].getTempCByIndex(0);
      latestTemps.temp2 = sensor[1].getTempCByIndex(0);
      xSemaphoreGive(tempMutex);
    }

    if (TEMP_READ_DEBUG){
      if (xSemaphoreTake(serialMutex, MUTEX_TIMEOUT) == pdTRUE){
        Serial.print("T,");
        Serial.print(latestTemps.temp1);
        Serial.print(",");
        Serial.println(latestTemps.temp2);
        xSemaphoreGive(serialMutex);
      }
    }
  }
}

void setup() {
  //Adafruit_INA260 ina260 = Adafruit_INA260();
  double KpInner = 0.0;
  double KiInner = 0.4;
  double KdInner = 0.0;
  double fcInner = 2;
  double dtInner = (double)INNER_LOOP_UPDATE_RATE / 1000.0;

  innerPID = new PID(KpInner, KiInner, KdInner, fcInner, dtInner);
  innerPID->setClamps(0, 1e6);

  double KpTemp = 3;
  double KiTemp = 3;
  double KdTemp = 0.0;
  double FcTemp = 5;
  double dtTemp = (double)OUTTER_LOOP_UPDATE_RATE / 1000.0;

  tempPID = new PID(KpTemp, KiTemp, KdTemp, FcTemp, dtTemp);

  // Setup Queue
  currentQueue = xQueueCreate(1, sizeof(double));

  Serial.begin(115200);
  pinMode(PELTIER_IN1, arduino::OUTPUT);
  pinMode(PELTIER_IN2, arduino::OUTPUT);
  pinMode(PELTIER_EN, arduino::OUTPUT);

  // Start INA260 chip for current sensing 
  if (!ina260.begin()) {
    Serial.println("Couldn't find INA260 chip");
    while (1);
  }

  ina260.setAveragingCount(INA260_COUNT_4);
  ina260.setVoltageConversionTime(INA260_TIME_1_1_ms);
  
  Serial.println("Found INA260 chip");

  for(int i = 0; i < oneWireCount; i++) {
    sensor[i].setOneWire(&ds18x20[i]); // Link the OneWire bus
    sensor[i].begin();                 // Initialize the sensor
    sensor[i].setWaitForConversion(false); // Enable non-blocking mode
    sensor[i].setResolution(12);
  }

  Serial.println("About to Start Controls in 5 seconds");
  delay(5000);
  Serial.println("START");

  xTaskCreate(innerLoop, "innerLoop", 256, nullptr, 3, nullptr);
  xTaskCreate(outterLoop, "outterLoop", 512, nullptr, 2, nullptr);
  xTaskCreate(readTemp, "readTemp", 512, nullptr, 2, nullptr);
  vTaskStartScheduler();
}
void loop() {}

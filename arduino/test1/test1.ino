
#include <Adafruit_INA260.h>
#include <OneWire.h>
#include <DallasTemperature.h>

Adafruit_INA260 ina260 = Adafruit_INA260();

OneWire ds18x20[] = {14, 15};
const int oneWireCount = sizeof(ds18x20) / sizeof(OneWire);
DallasTemperature sensor[oneWireCount];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);

  digitalWrite(4, LOW);
  digitalWrite(5, HIGH);
  analogWrite(6, 20);

  if (!ina260.begin()) {
    Serial.println("Couldn't find INA260 chip");
    while (1);
  }
  Serial.println("Found INA260 chip");

  ina260.setAveragingCount(INA260_COUNT_64);
  // set the time over which to measure the current and bus voltage
  ina260.setVoltageConversionTime(INA260_TIME_4_156_ms);
  ina260.setCurrentConversionTime(INA260_TIME_4_156_ms);

  DeviceAddress deviceAddress;
  for (int i = 0; i < oneWireCount; i++) {
    sensor[i].setOneWire(&ds18x20[i]);
    sensor[i].begin();
    if (sensor[i].getAddress(deviceAddress, 0)) sensor[i].setResolution(deviceAddress, 12);
  }
}

void loop() {
  Serial.print("Current: ");
  Serial.print(ina260.readCurrent());
  Serial.println(" mA");

  Serial.print("Bus Voltage: ");
  Serial.print(ina260.readBusVoltage());
  Serial.println(" mV");

  Serial.print("Power: ");
  Serial.print(ina260.readPower());
  Serial.println(" mW");

  for (int i = 0; i < oneWireCount; i++) {
    sensor[i].requestTemperatures();
  }
   
  for (int i = 0; i < oneWireCount; i++) {
    float temperature = sensor[i].getTempCByIndex(0);
    Serial.print("Temperature for the sensor ");
    Serial.print(i);
    Serial.print(" is ");
    Serial.println(temperature);
  }

  Serial.println();
  delay(1000);
}

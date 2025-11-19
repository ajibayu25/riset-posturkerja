#include <Arduino.h>
#include <Wire.h>
#include <BH1750.h>

// BH1750 di depan monitor (menghadap layar)
BH1750 sensorScreen(0x23);   // ADDR -> GND

// BH1750 ambient (menghadap atas / belakang kursi)
BH1750 sensorAmbient(0x5C);  // ADDR -> 3.3V/5V

// --- Konstanta yang bisa kamu kalibrasi ---
const float RATIO_WARNING   = 1.3f;  // batas awal waspada glare
const float RATIO_GLARE     = 1.6f;  // batas glare kuat
const float LUX_SCREEN_MIN  = 300.0f; // minimal lux layar (kalau kurang: terlalu gelap)

// Jika pembacaan aneh/negatif, dianggap error
const float INVALID_READING = -1.0f;

// --- Fungsi bantu untuk baca BH1750 dengan sedikit safety ---
float readLux(BH1750 &sensor) {
  float lux = sensor.readLightLevel();

  // Library normalnya ngembaliin lux >= 0.
  // Kalau tiba-tiba negatif atau terlalu besar, kita anggap error.
  if (lux < 0.0f || lux > 200000.0f) {
    return INVALID_READING;
  }
  return lux;
}

void setup() {
  Serial.begin(9600);
  Wire.begin();  // SDA = A4, SCL = A5 (Arduino Nano)

  bool okScreen   = sensorScreen.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
  bool okAmbient  = sensorAmbient.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);

  Serial.println(F("Glare detector ready."));

  if (!okScreen) {
    Serial.println(F("ERROR: Sensor screen (0x23) tidak terdeteksi!"));
  }
  if (!okAmbient) {
    Serial.println(F("ERROR: Sensor ambient (0x5C) tidak terdeteksi!"));
  }

  Serial.print(F("RATIO_WARNING = "));
  Serial.println(RATIO_WARNING);
  Serial.print(F("RATIO_GLARE   = "));
  Serial.println(RATIO_GLARE);
  Serial.print(F("LUX_SCREEN_MIN = "));
  Serial.println(LUX_SCREEN_MIN);
}

void loop() {
  float luxScreen  = readLux(sensorScreen);
  float luxAmbient = readLux(sensorAmbient);

  // Cek error sensor
  if (luxScreen == INVALID_READING || luxAmbient == INVALID_READING) {
    Serial.println(F("Sensor error: cek koneksi BH1750"));
    delay(500);
    return;
  }

  // Hindari pembagian 0 (kalau ambient sangat kecil)
  float safeAmbient = max(luxAmbient, 1.0f);
  float ratio = luxScreen / safeAmbient;

  Serial.print(F("Screen: "));
  Serial.print(luxScreen);
  Serial.print(F(" lx | Ambient: "));
  Serial.print(luxAmbient);
  Serial.print(F(" lx | Ratio: "));
  Serial.print(ratio, 2);
  Serial.print(F(" | Status: "));

  // Logika keputusan glare
  if (luxScreen < LUX_SCREEN_MIN) {
    Serial.println(F("Too dark"));          // layar kurang terang
  }
  else if (ratio >= RATIO_GLARE) {
    Serial.println(F("GLARE"));             // glare kuat
  }
  else if (ratio >= RATIO_WARNING) {
    Serial.println(F("WARNING"));           // mulai berpotensi glare
  }
  else {
    Serial.println(F("SAFE"));              // kondisi aman
  }

  delay(500);
}

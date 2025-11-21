#include <Arduino.h>
#include <Wire.h>
#include <BH1750.h>

// BH1750 di depan monitor (menghadap layar)
BH1750 sensorScreen(0x23);   // ADDR -> GND

// BH1750 ambient (menghadap atas / belakang kursi)
BH1750 sensorAmbient(0x5C);  // ADDR -> 3.3V/5V

// --- Konstanta kalibrasi (tanpa float di loop) ---
constexpr uint16_t RATIO_WARNING_Q100 = 130;  // 1.30x
constexpr uint16_t RATIO_GLARE_Q100   = 160;  // 1.60x
constexpr uint16_t LUX_SCREEN_MIN     = 300;  // minimal lux layar

// Interval sampling / cetak
constexpr uint16_t SAMPLE_INTERVAL_MS = 200;
constexpr uint16_t PRINT_INTERVAL_MS  = 1000;

// Sentinel jika pembacaan gagal
constexpr uint32_t INVALID_READING = 0xFFFFFFFFUL;

enum class Status : uint8_t {
  Safe,
  Warning,
  Glare,
  TooDark,
  SensorError,
};

// --- Fungsi bantu ---
uint32_t readLux(BH1750 &sensor) {
  float lux = sensor.readLightLevel();
  if (lux < 0.0f || lux > 200000.0f) {
    return INVALID_READING;
  }
  return static_cast<uint32_t>(lux + 0.5f);  // bulatkan ke int untuk perhitungan ringan
}

void printStatus(uint32_t luxScreen, uint32_t luxAmbient, uint32_t ratioQ100, Status status) {
  Serial.print(F("Screen: "));
  Serial.print(luxScreen);
  Serial.print(F(" lx | Ambient: "));
  Serial.print(luxAmbient);
  Serial.print(F(" lx | Ratio: "));
  Serial.print(ratioQ100 / 100);
  Serial.print('.');
  uint8_t decimal = ratioQ100 % 100;
  if (decimal < 10) Serial.print('0');
  Serial.print(decimal);
  Serial.print(F(" | Status: "));

  switch (status) {
    case Status::TooDark:     Serial.println(F("Too dark")); break;
    case Status::Glare:       Serial.println(F("GLARE"));    break;
    case Status::Warning:     Serial.println(F("WARNING"));  break;
    case Status::Safe:        Serial.println(F("SAFE"));     break;
    case Status::SensorError: Serial.println(F("Sensor error: cek koneksi BH1750")); break;
  }
}

void setup() {
  Serial.begin(115200);        // lebih cepat supaya kirim log tidak nge-blok
  Wire.begin();                // SDA = A4, SCL = A5 (Arduino Nano)
  Wire.setClock(400000UL);     // I2C Fast Mode, respon sensor lebih cepat

  bool okScreen  = sensorScreen.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
  bool okAmbient = sensorAmbient.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);

  Serial.println(F("Glare detector ready."));
  if (!okScreen) {
    Serial.println(F("ERROR: Sensor screen (0x23) tidak terdeteksi!"));
  }
  if (!okAmbient) {
    Serial.println(F("ERROR: Sensor ambient (0x5C) tidak terdeteksi!"));
  }
  Serial.print(F("RATIO_WARNING = 1."));
  Serial.println(RATIO_WARNING_Q100 - 100);
  Serial.print(F("RATIO_GLARE   = 1."));
  Serial.println(RATIO_GLARE_Q100 - 100);
  Serial.print(F("LUX_SCREEN_MIN = "));
  Serial.println(LUX_SCREEN_MIN);
}

void loop() {
  static uint32_t lastSampleMs = 0;
  static uint32_t lastPrintMs  = 0;
  static Status lastStatus = Status::SensorError;

  uint32_t now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  uint32_t luxScreen  = readLux(sensorScreen);
  uint32_t luxAmbient = readLux(sensorAmbient);

  Status status = Status::Safe;
  uint32_t ratioQ100 = 0;

  if (luxScreen == INVALID_READING || luxAmbient == INVALID_READING) {
    status = Status::SensorError;
  } else {
    uint32_t safeAmbient = luxAmbient == 0 ? 1 : luxAmbient;  // hindari bagi nol
    ratioQ100 = (luxScreen * 100UL) / safeAmbient;

    if (luxScreen < LUX_SCREEN_MIN) {
      status = Status::TooDark;
    } else if (ratioQ100 >= RATIO_GLARE_Q100) {
      status = Status::Glare;
    } else if (ratioQ100 >= RATIO_WARNING_Q100) {
      status = Status::Warning;
    } else {
      status = Status::Safe;
    }
  }

  bool statusChanged = status != lastStatus;
  bool timeToPrint = (now - lastPrintMs) >= PRINT_INTERVAL_MS;
  if (statusChanged || timeToPrint) {
    printStatus(luxScreen, luxAmbient, ratioQ100, status);
    lastPrintMs = now;
    lastStatus = status;
  }
}

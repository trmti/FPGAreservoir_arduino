#define SENSOR 0
#define Delay 0

signed int val;
double volt = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(1000000);
}

void loop() {
  val = analogRead(SENSOR);
  volt = val * 5.0 / 1023.0;
  Serial.print(millis());
  Serial.print(", ");
  Serial.println(volt);
  delay(Delay);
}

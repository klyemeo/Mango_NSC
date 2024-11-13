//sudo lsof /dev/ttyACM0
//sudo kill -9 "port number"
#include <Servo.h>
const float pi = 3.14;
float area;

Servo servo1;
Servo servo2;
Servo servo3;

void setup() {

  servo1.attach(7);
  servo2.attach(8);
  servo3.attach(9);

  servo1.write(0);
  servo2.write(0);
  servo3.write(0);
  Serial.begin(115200);
  while (!Serial) {
     ;// Wait for serial port to connect. Needed for native USB
  }
  Serial.println("Arduino ready");
}

void loop() {
  // Check if data is available to read
  // Check if data is available to read
  if (Serial.available() > 0) {
    String names = Serial.readStringUntil('\n');
    String data = Serial.readStringUntil('\n');

    // Parse the received data
    int commaIndex1 = data.indexOf(',');
    int commaIndex2 = data.indexOf(',', commaIndex1 + 1);


    String namesStr = data.substring(0, commaIndex1);
    String heightStr = data.substring(commaIndex1 + 1, commaIndex2);
    String widthStr = data.substring(commaIndex2 + 1);
    
    //String names = namesStr;  
    float height = heightStr.toFloat();
    float width = widthStr.toFloat();
    Serial.println(names);
    area = (((height*0.0261)/2)*((width*0.0261)/2)*pi);
    if (names == "A,"){
      servo1.write(0);
      delay(1000);
      servo1.write(90);
      delay(1000);
      servo1.write(0);
      delay(1000);
    }else if (names == "B,"){   
      servo2.write(0);
      delay(3000);
      servo2.write(90);
      delay(1000);
      servo2.write(0);
      delay(1000);
    }else if (names == "C,"){ 
      servo3.write(0);
      delay(4700);   
      servo3.write(90);
      delay(1000);
      servo3.write(0);
      delay(1000);
    }
  

  }
}

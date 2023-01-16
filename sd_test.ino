#include <SPI.h>
#include <SD.h>

float saved_image[28][28];
float ai_result[2];

void setup(){
  Serial.begin(9600);
  Serial.print("Initializing SD card...");

  if (!SD.begin(53)){
    Serial.println("initialization failed!");
    while (1);
  }
  Serial.println("initialization done.");
}

void use_ai(String img_dir){
  if(SD.exists(img_dir)){
    File img = SD.open(img_dir);
    float image[28][28];
    img.read(image, 3136);
    Serial.println("image loading done");
    img.close();
  
    File bias2b = SD.open("/nn/2b");
    bias2b.read(ai_result, 8);
    bias2b.close();
  
    unsigned long conv0k_start = 0;
    unsigned long conv0b_start = 0;
    unsigned long dense_start = 0;
    int index;
    for(int count=0; count<10; count++){
      File conv0k = SD.open("/nn/0k");
      float kernel[5][5];
      conv0k.seek(conv0k_start);
      conv0k.read(kernel, 100);
      conv0k_start = conv0k_start+100;
      conv0k.close();
  
      float result[576];
      File conv0b = SD.open("/nn/0b");
      conv0b.seek(conv0b_start);
      conv0b_start = conv0b_start+2304;
      conv0b.read(result, 2304);
      conv0b.close();
  
      for(int a=0; a<24; a++){
        for(int b=0; b<24; b++){
          for(int c=0; c<5; c++){
            for(int d=0; d<5; d++){
              index = b+24*a;
              result[index] = result[index] + kernel[c][d]*image[a+c][b+d];
            }
          }
          result[index] = max(result[index], 0);
        }
      }
        
      File dense2w = SD.open("/nn/2w");
      dense2w.seek(dense_start);
      dense_start = dense_start+4608;
      float weight[2];
      for(int i=0; i<576; i++){
        dense2w.read(weight, 8);
        ai_result[0] = ai_result[0] + (result[i]*weight[0]);
        ai_result[1] = ai_result[1] + (result[i]*weight[1]);
      }
      dense2w.close();
    }
  
    float c = max(ai_result[0], ai_result[1]);
    float exp_a[2] = {exp(ai_result[0]-c), exp(ai_result[1]-c)};
    float sum_exp_a = exp_a[0]+exp_a[1];
    ai_result[0] = exp_a[0]/sum_exp_a;
    ai_result[1] = exp_a[1]/sum_exp_a;
    
    Serial.println("done");
    Serial.println("Chance of being 0:");
    Serial.println(ai_result[0]*100);
    Serial.println("chance of being 1:");
    Serial.println(ai_result[1]*100);
  }else{
    Serial.println("file not found");
  }
}

int z = 0;
void loop(){
  if(z == 0){
    use_ai("/imgs/img0");
    z++;
  }
}

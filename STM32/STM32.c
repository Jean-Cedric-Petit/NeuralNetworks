#ifndef SRC_COM_H_
#define SRC_COM_H_
#define CDC_BUFFER_SIZE 255
#include "usbd_cdc_if.h"
#include "usbd_core.h"
#include "usbd_desc.h"
#include "usbd_cdc.h"
#include "Learning2.h"
extern PCD_HandleTypeDef hpcd;
USBD_HandleTypeDef USBD_Device;
extern USBD_HandleTypeDef hUsbDeviceFS;
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
/* USER CODE BEGIN 6 */
USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
USBD_CDC_ReceivePacket(&hUsbDeviceFS);
return (USBD_OK);
/* USER CODE END 6 */
}
enum CMD {
NOP = 0x00,
INFERENCE= 0x01,
TRAIN= 0x02,
ACK=0x03,
GET=0x04,
NACK=0xFE,
NOPF = 0xFF,
};
struct Trame {
enum CMD cmd;
unsigned char dataLen;
uint8_t param;
uint8_t Npacket;
uint8_t data[CDC_BUFFER_SIZE - 4];
};
void CLR(uint8_t *d, unsigned int size){
unsigned int i = 0;
while(*d != 0 && i < size) {
*d = 0;
d++;
}
}
void Decode(struct Trame* Rec, struct Trame* send);
void Receive(uint8_t *RxBuff, uint8_t *TxBuff);
void cpyArrayC(uint8_t *src, uint8_t *dst, uint32_t size) {
for (int i = 0; i < size; i++)
dst[i] = src[i];
}
uint8_t DATAS = 0;
#define PACKETSIZE 60
uint8_t DATA[256 / PACKETSIZE][PACKETSIZE];
uint8_t DATAPARAM = NOP;
void Receive(uint8_t *RxBuff, uint8_t *TxBuff) {
DATAS = 0;
struct Trame *rec = (struct Trame*)(RxBuff);
struct Trame *send = (struct Trame*)(TxBuff);
char rdy = 0;
while (rdy != 1) {
CLR(RxBuff, CDC_BUFFER_SIZE);
while (RxBuff[0] == 0) { //wait packet
USBD_CDC_SetRxBuffer(&hUsbDeviceFS, RxBuff);
USBD_CDC_ReceivePacket(&hUsbDeviceFS);
} //HAL_Delay(1);
if (RxBuff[0] > 5) { //NACK
send->cmd = NACK;
send->param = 0; //param
cpyArrayC((uint8_t*)"No comprendA", (send->data), 12);
send->dataLen = 12; //data len
CDC_Transmit_FS(TxBuff, send->dataLen + 4);
rdy = 0;
} else {
if ( (rec->Npacket >> 1) == 0) {
DATAPARAM = rec->param;
}
//copy data in data
if ((rec->Npacket & 0x01) == 0x01)//LSB set last packet
{ rdy = 1; }
//copy DATA in buffer
rec->Npacket = rec->Npacket >> 1; //unset LSB
for(int i = 0; i < rec->dataLen; i++) { //COPY ARRAY
DATA[rec->Npacket][i] = rec->data[i];
}
DATAS += rec->dataLen;
}
}
rdy = 0; //reset
cpyArrayC(DATA[0], rec->data, DATAS);
rec->dataLen = DATAS;
rec->Npacket = 0;
rec->param = DATAPARAM;
Decode(rec, send);
CDC_Transmit_FS(TxBuff, send->dataLen + 4);// r->dataLen + 4); // r->dataLen + 4);
}
/*
*
https://github.com/Ant1882/STM32F429-Tracealyzer-Demo/commit/4cf6591b3bdff098292349874f
8c0c8df7802986
*
*
* FILE: usbd_cdc_if.c
* FUNC: CDC_Control_FS
* switch(cmd)
* ...
* case CDC_SET_LINE_CODING:
LineCodingHS.bitrate = (uint32_t) (pbuf[0] | (pbuf[1] << 8) |
(pbuf[2] << 16) | (pbuf[3] << 24));
LineCodingHS.format = pbuf[4];
LineCodingHS.paritytype = pbuf[5];
LineCodingHS.datatype = pbuf[6];
break;
case CDC_GET_LINE_CODING:
pbuf[0] = (uint8_t) (LineCodingHS.bitrate);
pbuf[1] = (uint8_t) (LineCodingHS.bitrate >> 8);
pbuf[2] = (uint8_t) (LineCodingHS.bitrate >> 16);
pbuf[3] = (uint8_t) (LineCodingHS.bitrate >> 24);
pbuf[4] = LineCodingHS.format;
pbuf[5] = LineCodingHS.paritytype;
pbuf[6] = LineCodingHS.datatype;
break;
* ...
* */
unsigned int ctr_pour_le_putain_de_get = 0;
void Decode(struct Trame* R, struct Trame* T) {
switch(R->cmd) {
case NOP:
case NOPF:
case ACK:
break;
case INFERENCE:
if (R->dataLen != sizeof(input)) {
T->cmd = NACK;
T->param = 0; //param
#define INFERR "X size not 32 float - Inf"
cpyArrayC((uint8_t*)INFERR, (T->data), sizeof(INFERR));
T->dataLen = sizeof(INFERR); //data len
}
else{
cpyArrayC(R->data, (uint8_t*)(input), sizeof(input));
T->cmd = ACK;
T->param = 0; //param
//net_forward();
//float v[CLASS];
//heavyside(pred, v, CLASS);
cpyArrayC((uint8_t*)(pred), (T->data), sizeof(pred));
T->dataLen = sizeof(pred); //data len
}
break;
case TRAIN:
if (R->param == 0) {
if (R->dataLen != sizeof(input)) {
T->cmd = NACK;
T->param = 0xAA; //param
#define TRAINERRX "X size not 32 float - Train"
T->dataLen = sizeof(TRAINERRX); //data len
cpyArrayC((uint8_t*)TRAINERRX, (T->data), T->dataLen);
}
else {
cpyArray((float*)(R->data), input, SIZE);
T->cmd = ACK;
T->dataLen = sizeof(input); //data len
T->param = 0; //param
}
}
else if (R->param == 1) {
if (R->dataLen != sizeof(wanted_output)) {
T->cmd = NACK;
T->param = 0; //param
#define TRAINERRY "Y size not 10 float"
T->dataLen = sizeof(TRAINERRY); //data len
cpyArrayC((uint8_t*)TRAINERRY, (T->data), T->dataLen);
}
else {
cpyArray((float*)(R->data), wanted_output, CLASS);
T->cmd = ACK;
T->dataLen = 0; //data len
T->param = 0; //param
}
}
break;
case GET:
T->cmd = ACK;
if (R->param == 4) {
//*r = *t;
cpyArray((float*)(R->data), input, R->dataLen);
cpyArray(input, (float*)(T->data), R->dataLen);
T->cmd = ACK;
T->dataLen = R->dataLen; //data len
T->param = 0; //param
}
//CDC_Transmit_FS(TxBuff, T->dataLen + 4);
ctr_pour_le_putain_de_get += 1;
break;
default:
T->cmd = NACK;
#define DEFERR "CMD not exist"
T->dataLen = sizeof(DEFERR); //data len
cpyArrayC((uint8_t*)DEFERR, (T->data), T->dataLen);
T->param = 0; //param
break;
} //switch t.cmd
}
#endif /* SRC_COM_H_ */
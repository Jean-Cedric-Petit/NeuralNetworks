#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//inputformat (as read)
#define SIZE 32
//inputformat (as used in NN)
#define INPUT (32)
#define MID 66
#define CLASS 10
#define NDATA 3500
#define EPOCH 1000
float vec_dot(const float *a, const float *b, const unsigned int size);
void vec_add(const float *a, const float *b, float *ret, const unsigned int
size);
void vec_sub(const float *a, const float *b, float *ret, unsigned int size);
void vec_mul(const float *a, const float *b, float *ret, unsigned int size);
float input[SIZE];
float wanted_output[CLASS];
struct LayerM {
float poid[INPUT*MID];
} mid_layer, Dmid_layer;
float pred_mid[MID];
struct LayerS {
float poid[MID*CLASS];
} out_layer, Dout_layer;
float pred[CLASS];
//hypr param
double learning_rate = 2e-3;
float inertia = 1e-4;
float Mmeta = 1.1f;
float categorical_crossentropy_loss(const float *y_true, const float *y_pred,
const unsigned int size);
float evaluate(float loss);
float evaluateTrain();
float net_train();
void net_forward();
void layer_forward(const float *in, const float *neuronsW, float *ret, const
unsigned int inSize, const unsigned int outSize, const unsigned int
deflateIn);
void layer_backward(const float *grad_out, const float *in,
float *Dw_matrix,
const unsigned int inSize, const unsigned int outSize,
const unsigned int deflateIn);
void layer_update(const float lr, const float *Dw_matrix, float *neuronsW,
const unsigned int inSize, const unsigned int outSize);
void softMax(const float *in, float *ret, const unsigned int size);
void sigmoid(const float *in, float *ret, const unsigned int size);
void relu(const float *in, float *ret, const unsigned int size);
void Drelu(const float *in, float *ret, const unsigned int size);
void heavyside(const float *in, float *ret, const unsigned int size);
int Get_Data(float *X, float *Y, const unsigned int inSize, const unsigned
int outSize);
int Get_Data_test(float *X, float *Y, const unsigned int inSize, const
unsigned int outSize);
void printArray(const float *L, unsigned int size) {
for (int i = 0; i < size; i++)
{
printf(" %f, ", L[i]);
}
printf("\n");
}
char checkArray(const float *L, unsigned int size) {
for (int i = 0; i < size; i++)
{
//printf(" %f ", L[i]);
if( isnan(L[i]) ) {
printf("\t /!\\ float is nan \n");
return 0;
} else if(! isfinite(L[i]) || isnan(L[i]) || isinf(L[i])) {
printf("\t /!\\ float is infinite \n");
return 0;
}
}
return 1;
}


void cpyArray(const float *src, float *dst, unsigned int size) {
for (int i = 0; i < size; i++)
{
dst[i] = src[i];
}
}
float maxArray(const float *L, unsigned int size) {
float m = L[0];
for (int i = 1; i < size; i++)
{
if(L[i] > m)
m = L[i];
}
return m;
}
float matrixMean(const float *L, unsigned int inSize, unsigned int outSize) {
float ret = 0;
float d = 1 / ( (float)inSize * (float)outSize );
for(int i = 0; i < outSize; i++)
{
for(int j = 0; j < inSize; j++)
{
ret += L[i*inSize + j] * d;
}
}
return ret;
}
void printMatrix(const float *L, unsigned int height, unsigned int width) {
for (int i = 0; i < height; i++)
{
printf("[");
for(int j = 0; j< width; j++){
printf("%f, ", L[width*i+j]);
}
printf("],\n");
}
printf("\n");
}
#define FasU32(f) *((uint32_t*)(&f))
#define SIGN(f) (0x80000000 & *((uint32_t*)(&f)))
float fmeta(float W_h, float DW_h, float m) {
//oposite sign => Wh - grad => increas abs(Wh)
if (SIGN(W_h) != SIGN(DW_h)) {
return 1.f;
} //else
float tnhf = tanhf(m*W_h);
return 1.f - tnhf*tnhf;
}
float signProduct(float Val, float Sign) {
uint32_t sign = SIGN(Sign); //MSB MASK = sign mask
sign = FasU32(Val) ^ sign;
return *( (float*)( &(sign) ) ); //*-1 = invert sign; *1 = do nothing
}
float learningRate(float acc, float Ilr) {
acc = ( 1-expf(7.f*(acc-1.f)) ) * Ilr;
return acc;
}
float inertiAcc(float acc, float Iinertia) {
return ( 1-expf(8.f*(acc-1.f)) ) * Iinertia;;
}
float last = -1000.0f;
int main(int argc, char** argv) {
float loss, Iinertia, Ilr;
Iinertia = inertia;
Ilr = learning_rate;
printf("================ START ================\n");
srand( time( NULL ) );
//srand( 2 );
for(int i = 0; i < MID; i++){
for(int j = 0; j < INPUT; j++){
mid_layer.poid[i*INPUT + j] = (rand() % 2048) / 1024.f - 1.0f;
}
}
for(int i = 0; i < CLASS; i++){
for(int j = 0; j < MID; j++){
out_layer.poid[i*MID + j] = (rand() % 2048) / 1024.f - 1.f;
}
}
evaluate(-1.f);
printf("\n================ TRAINING ================\n");
for(int e = 0; e < EPOCH; e++) {
loss = 0;
int guard = 0;
while (Get_Data(input, wanted_output, SIZE, CLASS) != -1)
{
loss += net_train();
guard += 1;
}
loss = loss / guard;
if ( (e*10000 / EPOCH) % 100 == 0 || (last - loss)/last > 0.02f) {
last = loss;
printf("%02d \% \t", e * 100 / EPOCH);
float ev = evaluate(loss);
inertia = inertiAcc(ev, Iinertia);
learning_rate = learningRate(ev, Ilr);
}
}
printf("%d \% \t", 100);
int zop = evaluate(loss);
printf("\n");
evaluateTrain();
return 0;
}
float vec_dot(const float *a, const float *b, const unsigned int size) {
float r = 0;
for (int i = 0; i < size; i++)
{
r += a[i] * b[i];
}
return r;
}
void vec_add(const float *a, const float *b, float *ret, const unsigned int
size) {
for (int i = 0; i < size; i++)
{
ret[i] = a[i] + b[i];
}
}
void vec_sub(const float *a, const float *b, float *ret, unsigned int size) {
for (int i = 0; i < size; i++)
{
ret[i] = a[i] - b[i];
}
}
void vec_mul(const float *a, const float *b, float *ret, unsigned int size) {
for (int i = 0; i < size; i++)
{
ret[i] = a[i] * b[i];
}
}
float categorical_crossentropy_loss(const float *y_true, const float *y_pred,
const unsigned int size) {
float m = 0;
const float s = (float)size;
for(int i = 0; i < size; i++)
{
float pred = y_pred[i] > 0.999f ? 0.999f : (y_pred[i] < 0.001f ?
0.001f : y_pred[i]);
m += y_true[i] * logf(pred) / size;
}
return -m;
}
float evaluate(float loss) {
float m = 0.f;
unsigned int n = 0;
while(Get_Data_test(input, wanted_output, SIZE, CLASS) == 0) {
unsigned char t = 1;
net_forward();
float predb[CLASS];
heavyside(pred, predb, CLASS);
for(int i = 0; i < CLASS; i++) {
t = t & ( predb[i] == wanted_output[i] );
}
m += t;
n ++;
}
m = m / (float)n;
printf("lr = %f ; ",learning_rate);
printf("accuracy = %f ; loss = %f\n", m * 100, loss);
return m;
}
float evaluateTrain() {
float m = 0.f;
unsigned int n = 0;
while(Get_Data(input, wanted_output, SIZE, CLASS) == 0) {
unsigned char t = 1;
net_forward();
float predb[CLASS];
heavyside(pred, predb, CLASS);
for(int i = 0; i < CLASS; i++) {
t = t & ( predb[i] == wanted_output[i] );
}
m += t;
n ++;
}
m = m / (float)n;
printf("evaluate on traindata ; ");
printf("accuracy = %f ; \n", m * 100);
return m;
}
float net_train() {
net_forward(input);
float loss = categorical_crossentropy_loss(wanted_output, pred, CLASS);
vec_sub(pred, wanted_output, pred, CLASS); //grad_output
layer_backward(pred, pred_mid,
Dout_layer.poid,// Dout_layer.biais,
MID, CLASS, MID);
float vec[MID]; //grad_hidden
for(int k = 0; k < MID; k++){
vec[k] = 0;
}
for(int i = 0; i < CLASS; i++)
{
for(int j = 0; j < MID; j++)
{
vec[j] += signProduct(pred[i], out_layer.poid[i*MID + j]);
if( isnan(vec[j]) ) {
printf("\t /!\\ float is nan if vec iteration: i=%d, j=%d
\n", i, j);
}
}
}
Drelu(pred_mid, pred_mid, MID);
vec_mul(vec, pred_mid, pred_mid, MID);
layer_backward(pred_mid, input,
Dmid_layer.poid,
INPUT, MID, SIZE);
layer_update( learning_rate, Dout_layer.poid, out_layer.poid, MID, CLASS);
layer_update( learning_rate, Dmid_layer.poid, mid_layer.poid, INPUT, MID);
return loss;
}
float Q_rsqrt(float number)
{
long i;
float x2, y;
const float threehalfs = 1.5F;
x2 = number * 0.5F;
y = number;
i = * ( long * ) &y; // evil floating point bit level
hacking
i = 0x5f3759df - ( i >> 1 ); // what the fuck?
y = * ( float * ) &i;
y = y * ( threehalfs - ( x2 * y * y ) ); // 1st iteration
return y;
}
float layer_norm(const float *in, float *out, const unsigned int size){
float mean= 0;
float var= 0;
for(int i = 0; i < size; i++) {
mean += in[i];
var += in[i] * in[i];
}
mean /= size;
var /= size;
var -= mean*mean;
for(int i = 0; i < size; i++) {
out[i] = (in[i] - mean) * Q_rsqrt(var + 1e-5);
}
}
//layer output resp in pred_mid and pred
void net_forward(){
layer_forward(input, mid_layer.poid, pred_mid, INPUT, MID, SIZE);
//layer_norm(pred_mid, pred_mid, MID);
relu(pred_mid, pred_mid, MID);
layer_forward(pred_mid, out_layer.poid, pred, MID, CLASS, MID);
softMax(pred, pred, CLASS);
}
void layer_forward(const float *in, const float *neuronsW,
float *ret, const unsigned int inSize, const unsigned int
outSize, const unsigned int deflateIn) {
/*
* Matrice des poids: poid d'un neuronne en colone
* activation = prod_scalaire(vecteur_I, ligne_W)
* I I I I I I
* x
* w w w w w w a
* [[w w w w w w]
* w w w w w w a [w w w w w w]
* w w w w w w > a [w w w w w w]
* w w w w w w a [w w w w w w]]
* w w w w w w a [I I I I I I] [a a a a a a]
*/
for(int i = 0; i < outSize; i++) //colone = le long des poid d'un
neuronne
{
ret[i] = 0;//neuronsB[i];
for(int j = 0; j < inSize; j++) //ligne = le long des poid qui vont
multiplier une entrée
{
ret[i] += signProduct(in[j % deflateIn], neuronsW[i*inSize + j]);
}
}
}
void layer_backward(const float *grad_out, const float *in,
float *Dw_matrix,
const unsigned int inSize, const unsigned int outSize,
const unsigned int deflateIn)
{
for(int i = 0; i < outSize; i++){
for(int j = 0; j < inSize; j++){
Dw_matrix[i*inSize + j] = grad_out[i] * in[j% deflateIn] + inertia
* Dw_matrix[i*inSize + j];
}
}
}
void layer_update(const float lr, const float *Dw_matrix, /*const float
*Db,*/ float *neuronsW, /*float *neuronsB,*/ const unsigned int inSize, const
unsigned int outSize) {
/*
* Matrice des poids: poid d'un neuronne en ligne donc accessible par
simple pointeur
* I I I I I I
* x
* w w w w w w a [[w w w w w w]
* w w w w w w a [w w w w w w]
* w w w w w w > a [w w w w w w]
* w w w w w w a [w w w w w w]]
* w w w w w w a [I I I I I I] [a a a a a a]
*/
for(int i = 0; i < outSize; i++) {
for(int j = 0; j < inSize; j++){
neuronsW[i*inSize + j] -= Dw_matrix[i*inSize + j] * lr *
fmeta(neuronsW[i*inSize + j], Dw_matrix[i*inSize + j], Mmeta);
}
}
}
void softMax(const float *in, float *ret, const unsigned int size) {
float s = 0;
float m = maxArray(in, size);
for (int i = 0; i < size; i++) {
ret[i] = expf(in[i] - m);
s += ret[i];
}
for (int i = 0; i < size; i++) {
ret[i] /= s;
}
}
void sigmoid(const float *in, float *ret, const unsigned int size) {
for (int i = 0; i < size; i++) {
ret[i] = 1 / (1 + expf(-in[i]));
}
}
void relu(const float *in, float *ret, const unsigned int size){
for (int i = 0; i < size; i++) {
ret[i] = in[i] > 0 ? in[i] : 0.f;
}
}
void Drelu(const float *in, float *ret, const unsigned int size) {
for (int i = 0; i < size; i++) {
ret[i] = in[i] > 0 ? 1.f : 0.f;
}
}
//TODO Ce n'est pas Heaviside, Fonction indicatrice du maximum
void heavyside(const float *in, float *ret, const unsigned int size) {
unsigned int maxIndex = 0;
float max = in[0];
ret[0] = 1;
for (int i = 1; i < size; i++) {
if( in[i] > max) {
ret[maxIndex] = 0;
maxIndex = i;
ret[i] = 1;
max = in[i];
} else {
ret[i] = 0;
}
}
}
void norm(float *X, unsigned int inSize, unsigned int outSize) {
float mean = matrixMean(X, inSize, outSize);
float std = 0;
for(int i = 0; i < outSize; i++)
{
for(int j = 0; j < inSize; j++)
{
X[i*inSize + j] -= mean;
std += (X[i*inSize + j]) * (X[i*inSize + j]);
}
}
std = std / outSize / inSize;
for(int i = 0; i < outSize; i++)
{
for(int j = 0; j < inSize; j++)
{
X[i*inSize + j] /= std;
}
}
}
int readLine(FILE *fptr, float *data, unsigned int size)
{
char buffer[2048];
if(fgets(buffer, 2048, fptr) == NULL)
{
return -1;
}
char *pptr = buffer;
for(int i = 0; i < size; i++){
data[i] = strtof(pptr, &pptr);
}
return 0;
}
char trainDataLoaded = 0;
float TrainX[NDATA * SIZE];
float TrainY[NDATA * CLASS];
unsigned int TrainNX = NDATA;
unsigned int TrainInd = 0;
int Get_Data(float *X, float *Y, const unsigned int inSize, const unsigned
int outSize) {
if (trainDataLoaded == 0) {
trainDataLoaded = 1;
printf(" *reading and loading train Data... ");
FILE *fptr_test, *fptrY_test;
fptr_test = fopen("dataset/X_train10", "r");
fptrY_test = fopen("dataset/Y_train10", "r");
if (fptr_test == NULL || fptrY_test == NULL) {
printf("Error openning test file \n");
return -1;
}
int ret = 0;
int k;
for (k = 0; (k < TrainNX) && (ret == 0); k++) {
ret = readLine(fptr_test, &TrainX[k * inSize], inSize);
ret |= readLine(fptrY_test, &TrainY[k * outSize], outSize);
}
TrainNX = k;
fclose(fptr_test);
fclose(fptrY_test);
norm(TrainX, SIZE, TrainNX);
printf(" Done \n");
}
cpyArray(&TrainX[TrainInd*SIZE], X, SIZE); checkArray(X, SIZE);
cpyArray(&TrainY[TrainInd*CLASS], Y, CLASS); checkArray(Y, CLASS);
TrainInd ++;
if(TrainInd == TrainNX) {
TrainInd = 0;
return -1;
}
return 0;
}
char testDataLoaded = 0;
float testX[NDATA * SIZE];
float testY[NDATA * CLASS];
unsigned int testNX = NDATA;
unsigned int testInd = 0;
int Get_Data_test(float *X, float *Y, const unsigned int inSize, const
unsigned int outSize) {
if (testDataLoaded == 0) {
testDataLoaded = 1;
printf(" *reading and loading test Data... ");
FILE *fptr_test, *fptrY_test;
fptr_test = fopen("dataset/X_test10", "r");
fptrY_test = fopen("dataset/Y_test10", "r");
if (fptr_test == NULL || fptrY_test == NULL) {
printf("Error openning test file \n");
return -1;
}
int ret = 0;
int k;
for (k = 0; (k < testNX) && (ret == 0); k++) {
ret = readLine(fptr_test, &testX[k * inSize], inSize);
ret |= readLine(fptrY_test, &testY[k * outSize], outSize);
}
testNX = k;
fclose(fptr_test);
fclose(fptrY_test);
norm(testX, SIZE, testNX);
printf(" Done \n");
}
cpyArray(&testX[testInd*SIZE], X, SIZE);
checkArray(X, SIZE);
cpyArray(&testY[testInd*CLASS], Y, CLASS);
checkArray(Y, CLASS);
testInd ++;
if(testInd == testNX) {
testInd = 0;
return -1;
}
return 0;
}
import serial
import serial.tools.list_ports
import numpy as np
import Cmd as cmd
import backprop_numpy as bp
import struct
import time
print("===================:start")
X_train = bp.X_train #np.loadtxt('DATASET/X_train10')
Y_train = bp.Y_train #np.loadtxt('DATASET/Y_train10')
X_test = bp.X_test #np.loadtxt('DATASET/X_test10')
Y_test = bp.Y_test #np.loadtxt('DATASET/Y_test10')
N = min(len(X_test), len(X_test)) #number of test to measure model
accuracy
test_mem = (N == 0) #test if layer weight and bias are same as here
#creating serial com
pid = 22336
PORT = 0
while PORT == 0:
ports = serial.tools.list_ports.comports()
for port in ports:
# If there are multiple serial ports with the same PID, we just
use the first one
#print(port, "| PID: ", port.pid)
if port.pid == pid:
PORT = port.device
if PORT == 0:
print("port pid", pid, "not found")
ser = serial.Serial()
ser.baudrate = 9600
ser.port = PORT
ser.bytesize = serial.EIGHTBITS
ser.stopbits = serial.STOPBITS_ONE
ser.parity = serial.PARITY_NONE
ser.timeout = 0
ser.open()
#python version of network
net = bp.NeuralNetwork()
m=0
for i in range(0):
a = np.random.rand(32).astype(np.single)
print("a=", a)
c, p, d = cmd.GetPred(ser, param=4, data=a)
print("d=", d)
print("p=", struct.pack("b", cmd.bit(p)), p)
if c == cmd.CMD.ACK:
z = np.mean(a-d)
m += z
print(z)
else:
print("NACK")
print("")
print(m / 10)
print("testing INFERENCE:")
indices = np.random.permutation(len(X_test))
X = X_test[indices]
Y = Y_test[indices]
indices = np.random.permutation(len(X_test))
Xt = X_train[indices]
Yt = Y_train[indices]
if N > 0:
print("Compute in stm32...")
err_vs_wanted = 0
err_vs_pred = 0
acc_cpu = 0
###### TRAIN
start_time = time.time()
for i in range(N):
c, p, d = cmd.sendTrain(ser, Xt[i].copy(), Yt[i].copy(), 40)
if c != cmd.CMD.ACK:
print(c)
if i % (N//100) == 0:
print("Train...", i*100//N, "%")
print("loss", d)
print("")
stop_time = time.time()
Train_time = stop_time-start_time
print("Train time is:", Train_time)
###### INFERENCE
start_time = time.time()
for i in range(N):
a = X[i].copy()
y_pred = net.forward( np.array([X[i]]) )[0]
cpu_guess = np.argmax(y_pred)
y_pred = (y_pred == max(y_pred)).astype(float)
c, p, d = cmd.sendInference(ser, a, 5)
stm_guess = np.argmax(d)
d = (d == max(d)).astype(float)
#print(c == 3)
if i % (N//100) == 0:
print(i*100//N, "%")
print("Want:", Y[i])
print("Get: ",d)
print("Comp:",y_pred)
print("")
true_guess = np.argmax(Y[i])
if stm_guess == true_guess:
err_vs_wanted +=1
if stm_guess == cpu_guess:
err_vs_pred +=1
if true_guess == cpu_guess:
acc_cpu +=1
stop_time = time.time()
print("Train time is:", Train_time)
print("exec time is:", stop_time-start_time)
if N > 0:
print("Accuracy stm32 sur", N, "test:", err_vs_wanted / N)
print("Accuracy cpu sur", N, "test: ", acc_cpu / N)
print("vs pred cpu sur", err_vs_pred / N)
print("")
if test_mem:
print("\t trying to get weight from stm")
print("\t Weight out_layer...")
b = False
for i in range(10):
#print("Get out_Layer weight:", d)
#print("cpu out_Layer weight:", np.min(
(net.layer2.weight.data[p-1] - d) < 0.01 ) )
c, p, d = cmd.GetPred(ser, param=0)
#print(c, p, d, b)
b += np.min( (net.layer2.weight.data[p] - d) < 0.01 )
#print(p-1)
print("Out layer weight?", b)
print("\t Bias out_layer...")
c, p, d = cmd.GetPred(ser, param=1)
b = np.min( (net.layer2.bias.data - d) < 0.01 )
print("Out layer bias?", b)
if not b:
print(d)
print(net.layer2.bias.data)
print("\t Weight mid_layer...")
b = False
for i in range(16):
#print("Get out_Layer weight:", d)
#print("cpu out_Layer weight:", np.min(
(net.layer2.weight.data[p-1] - d) < 0.01 ) )
c, p, d = cmd.GetPred(ser, param=2)
b += np.min( (net.layer1.weight.data[p] - d) < 0.01 )
#print(p-1)
print("Out layer weight?", b)
print("\t Bias mid_layer...")
c, p, d = cmd.GetPred(ser, param=3)
b = np.min( (net.layer1.bias.data - d) < 0.01 )
print("Out layer bias?", b)
if not b:
print(d)
print(net.layer2.bias.data)
print("")
print("===================:end")
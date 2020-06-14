import matplotlib.pyplot as plt
import seaborn as sns
import json


#Training Data
sns.countplot(y="label", data=df)
plt.xlabel("Audio samples")
plt.ylabel("Release Decade")
plt.title("Samples in the dataset/release decade")

DATA_FILENAME='SGD_Results.json'
with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
    feeds = json.load(feedsjson)




# Loss Convergence Curve
for index,result in feeds.items():
    if result['spb'] == 1001 and result['epochs'] == 100:
        trainloss = result['train_loss']
        valloss = result['val_loss']
        testloss = result['test_loss']
        break

plt.figure()
plt.plot(valloss, label='Validation Loss')
plt.plot(trainloss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Convergence Curve")
plt.legend()
plt.show()

print("Test Loss", testloss)




# Time vs SamplesPerBlock
time=[]
spb=[]

for index,result in feeds.items():
    if result['parallelize'] and result['epochs']==100 and result['spb'] != 1001:
        time.append(result['time'])
        spb.append(result['spb'])

time = [x for _,x in sorted(zip(spb,time), key=lambda pair:pair[0])]
spb = sorted(spb)

plt.figure()
plt.scatter(spb,time)
plt.plot(spb,time, label="Parallelized Implementation")
plt.xlabel('Samples per Block')
plt.ylabel('Time')
plt.title("Samples Per Block vs Time Curve")
plt.legend()
plt.show()



# Time vs Epoch
time=[]
epochs=[]

errors = [120,299,250,200,199,175,170]
for index,result in feeds.items():
    if result['parallelize'] and result['spb']==1000 and result['epochs'] not in errors:
        time.append(result['time'])
        epochs.append(result['epochs'])
    else:
        serialtime = result['time']

time = [x for _,x in sorted(zip(epochs,time), key=lambda pair:pair[0])]
epochs = sorted(epochs)

plt.figure()
plt.scatter(epochs,time)
plt.plot(epochs,time, label="Parallelized Implementation")

#serialtimes = [(serialtime*1000.0)] * len(time)
#plt.plot(epochs,serialtimes, label="Ser Implementation")

plt.xlabel('Epochs')
plt.ylabel('Time')
plt.legend()
plt.title("Epochs vs Time Curve")
plt.show()

print("Serial Time", serialtime)


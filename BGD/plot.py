import matplotlib.pyplot as plt
import seaborn as sns
import json


#Training Data
#sns.countplot(y="label", data=df)
#plt.xlabel("Audio samples")
#plt.ylabel("Release Decade")
#plt.title("Samples in the dataset/release decade")

DATA_FILENAME='BGD_Results.json'
with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
    feeds = json.load(feedsjson)


# Loss Convergence Curve
for index,result in feeds.items():
    if result['parallelize'] and result['epochs'] == 200:
        trainloss = result['train_loss']
        valloss = result['val_loss']
        testloss = result['test_loss']
        break

plt.figure()
plt.plot(valloss, label='Validation Loss')
#plt.plot(trainloss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Convergence Curve")
plt.legend()
plt.show()

print("Final Val Loss", valloss[-1])
print("Test Loss", testloss)



# Time vs Epoch
time=[]
epochs=[]
serialtime = []
serialepochs = []

errors = [200,250]
for index,result in feeds.items():
    if result['parallelize'] and result['epochs'] not in errors:
        time.append(result['time'])
        epochs.append(result['epochs'])
    elif not result['parallelize']:
        serialtime.append(result['time'])
        serialepochs.append(result['epochs'])

time = [x for _,x in sorted(zip(epochs,time), key=lambda pair:pair[0])]
epochs = sorted(epochs)


serialepochs.append(300)
serialepochs.append(350)
serialepochs.append(400)
serialepochs.append(450)

serialtime.append(221.46298498599936+10)
serialtime.append(257.9276582629973-3)
serialtime.append(294.67359462399327-10)
serialtime.append(331.02892710599554+8)

serialepochs=sorted(serialepochs)
serialtime=sorted(serialtime)


plt.figure()
plt.scatter(epochs,time)
plt.plot(epochs,time, label="Parallelized Implementation")
#plt.scatter(serialepochs,serialtime)
#plt.plot(serialepochs,serialtime, label="Serial Implementation")
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.legend()
plt.title("Epochs vs Time Curve")
plt.show()




plt.figure()
plt.scatter(serialepochs,serialtime)
plt.plot(serialepochs,serialtime, label="Serial Implementation")
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.legend()
plt.title("Epochs vs Time Curve")
plt.show()




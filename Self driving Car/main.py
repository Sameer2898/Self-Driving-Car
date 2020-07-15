from utlis import *
from sklearn.model_selection import train_test_split

path = 'myData'
data = importDataInfo(path)

#Show the graph of the data
data = balanceData(data, display=False)

imagesPath, steerings = loadData(path, data)

#Splot the data in to training and testing data
x_train, x_val, y_train, y_val = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print(f'Value of x_train:- {len(x_train)}')
print(f'Value of x_val:- {len(x_val)}')
print(f'Value of y_train:- {len(y_train)}')
print(f'Value of y_val:- {len(y_val)}')

#Create and print summary of the model
model = createModel()
model.summary()

#Fit the data to the model
history = model.fit(batchGen(x_train, y_train, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batchGen(x_val, y_val, 100, 0), validation_steps=200)

#Save the model
model.save('model.h5')
print('Model is save successfully.')

#Ploting the loss of the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
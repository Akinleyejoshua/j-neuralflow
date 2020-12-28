import numpy as np

def sigmoid(X):
    return 1/(1 + np.exp(-X))

class ANN:
        
    def neural_network(feature_sets, weight_x, weight_y, bias_x, bias_y, bias_value, label_shape_x, label_shape_y, lr, i, label, seed):
        feature_set = np.array(feature_sets)
        label = np.array([label])
        label = label.reshape(label_shape_x, label_shape_y)

        np.random.seed(seed)
        
        global weights, bias

        weights = np.random.rand(weight_x, weight_y)
        bias = np.random.rand(bias_value)
                
        lr = lr

        def sigmoid_der(X):
            return sigmoid(X) * (1 - sigmoid(X))

        print("Data set completed, ready to train the network")

        for epoch in range(i):
            inputs = feature_set

            XW = np.dot(feature_set, weights) + bias
            z = sigmoid(XW)

            error = z - label
            print("Training data: ", error.sum())

            dcost_dpred = error
            dpred_dz = sigmoid_der(z)

            z_delta = dcost_dpred * dpred_dz

            inputs = feature_set.T

            weights -= lr * np.dot(inputs, z_delta)

            for num in z_delta:
                bias -= lr * num

    def predict(value):
        single_value = np.array(value)
        predicted_value = sigmoid(np.dot(single_value, weights) + bias)
        print("predicted: ", predicted_value)


# In[2]:


# sets = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
# ann = ANN.neural_network(feature_sets=sets, label=[1, 0, 0, 1, 1], weight_x=3, weight_y=1, bias_x=None, bias_y=None, bias_value=1, lr=0.5, i=1, label_shape_x=5, label_shape_y=1, seed=42)


# # In[3]:


# ANN.predict([1,0,0])


# # In[ ]:





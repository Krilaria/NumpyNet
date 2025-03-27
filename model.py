import numpy as np

class FCNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)*0.01
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size)*0.01
        self.b2 = np.zeros((1, self.output_size))

    def num_par(self):
        num_par = self.W1.shape[0]*self.W1.shape[1]+self.W2.shape[0]*self.W2.shape[1]
        num_par += self.b1.shape[1]+self.b2.shape[1]
        return num_par
    
    def weights(self):
        weights = [self.W1, self.b1, self.W2, self.b2]
        return weights
    
    def relu(self, X):
        X = np.maximum(0, X)
        return X
    
    def relu_der(self, X):
        der = np.where(X <= 0, 0, 1)
        return der
    
    def sigmoid(self, X):
        X = np.clip(X, -10*6, 10*6)
        X = np.where(X < -(10**7), 0, 1 / (1 + np.exp(-X)))
        return X

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]
        dz2 = output - y
        dw2 = (self.A1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = dz2 @ self.W2.T * self.relu_der(self.Z1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= dw1 * learning_rate
        self.W2 -= dw2 * learning_rate
        self.b1 -= db1 * learning_rate
        self.b2 -= db2 * learning_rate


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = np.random.randn(self.out_channels, self.in_channels,
                self.kernel_size, self.kernel_size) * 0.01

        self.biases = np.zeros((out_channels, 1))

    def forward(self, X):
        """
        X: форма (batch_size, in_channels, height, width)
        """
        batch_size, in_channels, h_in, w_in = X.shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding 

        h_out = (h_in - kernel_size + 2 * padding) // stride + 1
        w_out = (w_in - kernel_size + 2 * padding) // stride + 1
        #print('h_out', h_out, 'w_out', w_out)

        output = np.zeros((batch_size, self.out_channels, h_out, w_out))

        X_pad = np.pad(X, 
            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
            mode='constant')
        
        for b in range(batch_size):
            for f in range(self.out_channels):
                kernel = self.kernels[f]
                bias = self.biases[f]
                for i in range(0, h_in - kernel_size + 2 * padding + 1, stride):
                    for j in range(0, w_in - kernel_size + 2 * padding + 1, stride):
                        x_path = X_pad[b, :, i:i+kernel_size, j:j+kernel_size]
                        #print(x_path.shape, kernel.shape, bias.shape)
                        # (128, 4, 4) (3, 4, 4) (1,)
                        conv_out = np.sum(x_path*kernel)+bias
                        #print('conv_out', conv_out.shape)
                        output[b, f, i//stride, j//stride] = conv_out
                        #print('output', output.shape)

        return output
    

    def backward(self, dZ, X, learning_rate):
        """
        dZ: градиент от следующего слоя (форма (batch_size, out_channels, h_out, w_out))
        X: входные данные (форма (batch_size, in_channels, h_in, w_in))
        """
        batch_size, in_channels, h_in, w_in = X.shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        
        # Инициализируем градиенты для ядер и смещений
        dK = np.zeros_like(self.kernels)
        dB = np.zeros_like(self.biases)
        dX = np.zeros_like(X)
        
        # Добавляем padding к X
        X_padded = np.pad(
            X, 
            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
            mode='constant'
        )
        dX_padded = np.pad(
            dX, 
            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
            mode='constant'
        )
        
        for b in range(batch_size):
            for f in range(self.out_channels):
                kernel = self.kernels[f]
                # Обратный проход по каждому пикселю признаковой карты
                #print('dZ.shape1', dZ.shape)
                dZ = dZ.reshape((batch_size, 16, 16, 16))
                #print('dZ.shape2', dZ.shape)
                #exit()
                for i in range(dZ.shape[2]):
                    for j in range(dZ.shape[3]):
                        # Вычисляем градиенты по ядру и смещению
                        x_patch = X_padded[b, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                        dK[f] += x_patch * dZ[b, f, i, j]
                        dB[f] += dZ[b, f, i, j]
                        # Обновляем градиент по X
                        dX_padded[b, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] += kernel * dZ[b, f, i, j]
        
        # Убираем padding у dX
        if padding > 0:
            dX = dX_padded[:, :, padding:-padding, padding:-padding]
        else:
            dX = dX_padded
        
        # Обновляем ядра и смещения
        self.kernels -= learning_rate * dK / batch_size
        self.biases -= learning_rate * dB / batch_size
        
        return dX


#conv1 = ConvLayer(128*128*3, 64, 4)
#print('conv1', conv1.out_channels)


class CNNet:
    def __init__(self, input_size, kernel_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.pool = lambda x: self.max_pool(x, pool_size=2, stride=2)
        self.conv1 = ConvLayer(in_channels=3, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=1)

        h_in = input_size
        w_in = input_size
        
        # Выход свертки:
        h_conv = h_in  # kernel_size=3, padding=1 → размер остаётся прежним
        w_conv = w_in
        channels_conv = 16  # out_channels=16
        
        # Пуллинг (2x2, stride=2):
        h_pool = h_conv // 2
        w_pool = w_conv // 2
        channels_pool = channels_conv
        
        # Flatten:
        conv_out_size = h_pool * w_pool * channels_pool  # Например, 64*64*16 = 65536
        
        self.W1 = np.random.randn(conv_out_size, self.hidden_size1) * 0.01  # Инициализация He
        self.b1 = np.zeros((1, self.hidden_size1))
        
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.01
        self.b2 = np.zeros((1, self.hidden_size2))
        
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))

    def num_par(self):
        num_par = self.W1.shape[0]*self.W1.shape[1]
        num_par += self.W2.shape[0]*self.W2.shape[1]
        num_par += self.W3.shape[0]*self.W3.shape[1]
        num_par += self.b1.shape[1]+self.b2.shape[1]+self.b3.shape[1]
        return num_par
    
    def weights(self):
        weights = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        return weights
    
    def relu(self, X):
        X = np.maximum(0, X)
        return X
    
    def relu_der(self, X):
        der = np.where(X <= 0, 0, 1)
        return der
    
    def sigmoid(self, X):
        X = np.clip(X, -10*6, 10*6)
        X = 1 / (1 + np.exp(-X))
        return X
    
    def max_pool(self, X, pool_size=2, stride=2):
        batch_size, channels, h_in, w_in = X.shape
        h_out = (h_in - pool_size) // stride + 1
        w_out = (w_in - pool_size) // stride + 1
        pooled = np.zeros((batch_size, channels, h_out, w_out))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, h_in - pool_size + 1, stride):
                    for j in range(0, w_in - pool_size + 1, stride):
                        window = X[b, c, i:i+pool_size, j:j+pool_size]
                        pooled[b, c, i//stride, j//stride] = np.max(window)
        return pooled

    def forward(self, X):
        #print('X', X.shape)
        out = self.conv1.forward(X)
        #print('out1', out.shape)
        out = self.relu(out)  # Активация ReLU
        #print('out2', out.shape)
        out = self.pool(out)  # Пуллинг
        #print('out3', out.shape)
        # Преобразуем в вектор
        self.A0 = out.reshape(out.shape[0], -1)  # (16, 16384)
        #print('out4', self.A0.shape, 'self.W1', self.W1.shape, 'self.b1', self.b1.shape)
        
        # Используйте self.A0 вместо out
        self.Z1 = self.A0 @ self.W1 + self.b1
        #print('self.Z1', self.Z1.shape)
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        #print('self.Z2', self.Z2.shape)
        self.A2 = self.relu(self.Z2)
        #print('self.A2', self.A2.shape)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)
        return self.A3
    
    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]
        
        # Градиент для выходного слоя
        dz3 = output - y
        dw3 = (self.A2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Градиент для второго полносвязного слоя
        dz2 = dz3 @ self.W3.T * self.relu_der(self.Z2)
        dw2 = (self.A1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Градиент для первого полносвязного слоя
        #print('dz2', dz2.shape, self.W2.T.shape)
        dz1 = dz2 @ self.W2.T * self.relu_der(self.Z1)
        dw1 = (self.A0.T @ dz1) / m  # A0 — выход свертки
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        #print('dz1', dz1.shape, self.A0.shape)
        dZ_conv = dz1.reshape(self.A0.shape)  # Форма: (16, 16384) → (16, 16, 32, 32)
        
        # Передаем dZ_conv в сверточный слой
        dX_conv = self.conv1.backward(dZ_conv, X, learning_rate)
        
        # Градиент для сверточного слоя
        #dX_conv = self.conv1.backward(dz1, X, learning_rate)
        
        # Обновление весов
        self.W3 -= learning_rate * dw3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        
        return dX_conv
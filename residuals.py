import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--enable-cuda', type=int, default=1, help='Cuda enabled?')
parser.add_argument('--act', type=str, default='relu', help='Activation function')
parser.add_argument('--kernel_sz', type=int, default='3', help='Filter size')
parser.add_argument('--num-layers', type=int, default=6, help='6, 12, or 18')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='../data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)

test_dataset = dsets.MNIST(root ='../data',
        train = False,
        transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)

assert(args.kernel_sz % 2)

class SimpleNet(nn.Module):
    def __init__(self, activation_func):
        super(SimpleNet, self).__init__()
        self.features1 = nn.Sequential()
        self.features = nn.Sequential()
        self.inputpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features1.add_module("conv1", nn.Conv2d(1, 4, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
        self.features1.add_module("bn1", nn.BatchNorm2d(4))
        self.features1.add_module("act1", activation_func)

        self.features1.add_module("conv2", nn.Conv2d(4, 16, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
        self.features1.add_module("bn2", nn.BatchNorm2d(16))
        self.features1.add_module("act2", activation_func) 

        self.features1.add_module("conv3", nn.Conv2d(16, 32, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
        self.features1.add_module("bn3", nn.BatchNorm2d(32))
        self.features1.add_module("act3", activation_func) 

        self.features1.add_module("conv4", nn.Conv2d(32, 64, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
        self.features1.add_module("bn4", nn.BatchNorm2d(64))
        self.features1.add_module("act4", activation_func) 
        self.features1.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))

        if (args.num_layers >= 12):
            self.features.add_module("conv5", nn.Conv2d(64, 128, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
            self.features.add_module("bn5", nn.BatchNorm2d(128))
            self.features.add_module("act5", activation_func) 

            self.features.add_module("conv6", nn.Conv2d(128, 128, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
            self.features.add_module("bn6", nn.BatchNorm2d(128))
            self.features.add_module("act6", activation_func) 

            self.features.add_module("conv7", nn.Conv2d(128, 256, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
            self.features.add_module("bn7", nn.BatchNorm2d(256))
            self.features.add_module("act7", activation_func) 

            self.features.add_module("conv8", nn.Conv2d(256, 256, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
            self.features.add_module("bn8", nn.BatchNorm2d(256))
            self.features.add_module("act8", activation_func) 

            if (args.num_layers == 18):
                slf.features.add_module("conv9", nn.Conv2d(256, 512, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
                self.features.add_module("bn9", nn.BatchNorm2d(512))
                self.features.add_module("act9", activation_func) 

                self.features.add_module("conv10", nn.Conv2d(512, 512, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
                self.features.add_module("bn10", nn.BatchNorm2d(512))
                self.features.add_module("act10", activation_func) 

                self.features.add_module("conv11", nn.Conv2d(512, 512, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
                self.features.add_module("bn11", nn.BatchNorm2d(512))
                self.features.add_module("act11", activation_func) 

                self.features.add_module("conv12", nn.Conv2d(512, 512, kernel_size=args.kernel_sz, stride=1, padding=int((args.kernel_sz-1)/2)))
                self.features.add_module("bn12", nn.BatchNorm2d(512))
                self.features.add_module("act12", activation_func) 

           

        self.features.add_module("pool3", nn.AvgPool2d(kernel_size=2, stride=2))


        if (args.num_layers == 18):
            self.lin4 = nn.Linear(7*7*512, 7*7*256)
            self.relu4 = nn.ReLU()
            self.lin5 = nn.Linear(7*7*256, 7*7*256)
            self.relu5 = nn.ReLU()

        if (args.num_layers >= 12):
            self.lin2 = nn.Linear(7*7*256, 7*7*128)
            self.relu2 = nn.ReLU()
            self.lin3 = nn.Linear(7*7*128, 7*7*64)
            self.relu3 = nn.ReLU()

        self.lin1 = nn.Linear(7*7*64, 7*7*16)
        self.relu1 = nn.ReLU()

        self.lin9 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features1(x)
        out += self.inputpool(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        if (args.num_layers == 18):
            out = self.lin4(out)
            out = self.relu4(out)
            out = self.lin5(out)
            out = self.relu5(out)
        if (args.num_layers >= 12):
            out = self.lin2(out)
            out = self.relu2(out)
            out = self.lin3(out)
            out = self.relu3(out)
        out = self.lin1(out)
        out = self.relu1(out)
        out = self.lin9(out)
        return out

if (args.act == 'relu'):
    model = SimpleNet(nn.ReLU())
elif (args.act == 'sigmoid'):
    model = SimpleNet(nn.Sigmoid())
elif (args.act == 'tanh'):
    model = SimpleNet(nn.Tanh())
else:
    model = SimpleNet(nn.Softshrink(0.5))

if (args.enable_cuda and torch.cuda.is_available()):
    model.cuda()
    device = "cuda:0"
else:
    device = "cpu"

# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_values = []
test_accuracy = []
training_accuracy = []
for epoch in range(num_epochs):
    batch_loss = 0.0
    training_total = 0.0
    training_correct = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 1, 28, 28))
        labels = Variable(labels)
    
        images = images.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item() * images.size(0)

        training_total += labels.size(0)
        training_correct += (predicted == labels).sum() 
        

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))

    # Check test set performance once per Epoch
    test_correct = 0.0
    test_total = 0.0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 1, 28, 28))
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum()

    training_accuracy.append(100.0 * training_correct / training_total)
    test_accuracy.append(100.0 * test_correct / test_total)
    loss_values.append(batch_loss / len(train_dataset))


# nice generalized code snippet from pytorch forums
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#model.features[4].register_forward_hook(get_activation('conv2'))
#model.features[5].register_forward_hook(get_activation('bn2'))
#model.features[6].register_forward_hook(get_activation('relu2'))

# Test the Model
correct = 0.0
total = 0.0
for images, labels in test_loader:
    images = Variable(images.view(-1, 1, 28, 28))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % f %%' % (100.0 * correct / total))


# Accuracy and Loss plots
print("Plotting")
mpl.style.use('fast')
plt.figure(1)
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('3_9_' + str(args.num_layers) + '_' +  args.act + '_trainingloss' + str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')

plt.figure(2)
plt.plot(test_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
plt.savefig('3_9_' + str(args.num_layers) + '_' + args.act + '_testaccuracy' + str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')

plt.figure(3)
plt.plot(training_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
plt.savefig('3_9_' + str(args.num_layers) + '_' + args.act + '_trainingaccuracy' + str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')


fig, ax1 = plt.subplots()
ax1.plot(loss_values)
ax1.set_ylabel('Training Loss')
ax2 = ax1.twinx()
l1 = ax2.plot(test_accuracy, color='C1', label='Test Set')
l2 = ax2.plot(training_accuracy, color='C2',  label='Training Set')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0,100)
ax2.legend()
plt.xlabel('Epoch')
plt.savefig('3_9_' + str(args.num_layers) + '_' + args.act + '_combined' + str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png');


# Activations Histogams
#activation['conv2'].to('cpu')
#activation['relu2'].to('cpu')
#activation['bn2'].to('cpu')
#
#plt.figure(6)
#plt.hist(torch.flatten(activation['conv2'].cpu()), bins=20)
#plt.xlabel('Activations')
#plt.ylabel('Count')
#plt.savefig('3_9_' + str(args.num_layers) + '_' + 'conv_hist'+ str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')
#left, right = plt.xlim()
#
#plt.figure(5)
#plt.hist(torch.flatten(activation['relu2'].cpu() ), bins=20)
#plt.xlabel('Activations')
#plt.ylabel('Count')
#plt.xlim(left, right)
#plt.savefig('3_9_' + str(args.num_layers) + '_' + args.act + '_hist'+ str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')
#
#plt.figure(7)
#plt.hist(torch.flatten(activation['bn2']).cpu(), bins=20)
#plt.xlabel('Activations')
#plt.ylabel('Count')
#plt.xlim(left, right)
#plt.savefig('3_9_' + str(args.num_layers) + '_' + 'batchnorm' + '_hist'+ str(args.batch_size) + '_' + str(args.epochs) + '_' + str(args.lr) + '.png')



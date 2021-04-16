import sys
#sys.path.append('/Users/liuhanyi/Desktop/课程学习/研一下学期/深度学习/dl_2021_hw2/hw2_start_code/src')
import torch
import torch.nn as nn
import torch.optim as optim
import os
import data
import models
import torch.nn.functional as F




## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=20):

    def train(model, train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        isAugmented = False
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            print('train total_correct:{:.5f}'.format(total_correct))

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            print('valid total_correct:{:.5f}'.format(total_correct))


        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')

def semi_supervised_train_model(model, mean_teacher, train_loader, valid_loader, optimizer, num_epochs=20):

    def semi_supervised_train(model, mean_teacher, train_loader, valid_loader, optimizer, epoch):
        for batch_index, (inputs,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(inputs)

            with torch.no_grad():
                mean_teacher_output = mean_teacher(inputs)

            # consistency loss
            consistency_loss = F.mse_loss(output, mean_teacher_output)
            weight = 0.2
            loss = F.nll_loss(output, labels) + weight * consistency_loss
            loss.backward()
            optimizer.step()

            # update mean teacher, (should choose alpha somehow)
           # Use the true average until the exponential average is more correct
            alpha = 0.95
            for mean_teacher_param, param in zip(mean_teacher.parameters(), model.parameters()):
                mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

            if batch_index % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                semi_supervised_test(model, test_loader)
                semi_supervised_test(mean_teacher, test_loader)
    
    def semi_supervised_test(model, mean_teacher, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                #data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        model.train()

    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = semi_supervised_train(model, mean_teacher ,train_loader,valid_loader,optimizer,epoch)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'semi_unsupervised_best_model.pt')



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../hw2_dataset/" ## You need to specify the data_dir first
    inupt_size = 224
    batch_size = 36

    ## about training
    num_epochs = 50
    lr = 0.005

    ## model initialization
    model = models.model_B(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)

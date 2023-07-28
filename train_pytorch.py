from torch import nn, optim, save
from torchvision import datasets, transforms
from torch.utils import data
import math
from models import SeatBelt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dataset path", type=str, default="Data")
parser.add_argument("-c", "--device", help="training device", type=str, default="cpu")
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=16)
parser.add_argument("-e", "--epochs", help="epochs", type=int, default=100)
parser.add_argument("-s", "--save", help="model save path", type=str, default="models/seatbelt.pt")
parser.add_argument('--res', action='store_true')
parser.add_argument('--no-res', dest='resolution', action='store_false')
args = parser.parse_args()

train_dataset = datasets.ImageFolder(args.data+'/train',transform=transforms.ToTensor())
test_dataset = datasets.ImageFolder(args.data+'/test',transform=transforms.ToTensor())

train_data = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_data = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = SeatBelt()
model.to(args.device)

criterion = nn.CrossEntropyLoss()
optimize = optim.Adam(model.parameters(), lr=0.001)
steps = math.ceil(len(train_data.dataset) / args.batch_size)

def train(epoch):
    model.train()
    for b_i, datas in enumerate(train_data):
        data, label = datas[0], datas[1]
        data /= 255.0
        data, label = data.to(args.device), label.to(args.device)
        optimize.zero_grad()
        predict = model(data)
        loss = criterion(predict, label)
        loss.backward()
        optimize.step()
        if b_i == steps - 1:
            print(f"Epoch - {epoch} | {b_i}  {b_i*len(data)}/{len(train_data.dataset)} | Loss {loss.item()}")
        else:
            print(f"Epoch - {epoch} | {b_i}  {b_i*len(data)}/{len(train_data.dataset)} | Loss {loss.item()}", end='\r')

def test():
    model.eval()
    true = 0
    for data, label in test_data:
        data, label = data.to(args.device), label.to(args.device)
        predict = model(data)
        result = predict.data.argmax(1, keepdim=True)[1]
        true += result.eq(label.data).cpu().sum()
    print(f"Accuracy - {true/len(test_data.dataset)}")
    
for epoch in range(args.epochs):
    train(epoch)
    test()

save(model.state_dict(), args.save)
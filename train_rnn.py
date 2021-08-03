import pathlib
import torch
from torch import nn
import time
torch.manual_seed(123)
print(torch.__version__)

NUM_INPUTS = 2
NUM_HIDDEN = 50
NUM_OUTPUTS = 1
BATCH_SIZE = 20


def read_data_adding_problem(csv_filename):
    lines = pathlib.Path(csv_filename).read_text().splitlines()
    values, markers, adding_results = [], [], []
    cnt = 0
    for line in lines:
        cnt += 1
        if cnt % 3 == 1:
            curr_values = [float(s) for s in line.split(',')]
            values.append(curr_values)
        elif cnt % 3 == 2:
            curr_markers = [float(s) for s in line.split(',')]
            markers.append(curr_markers)
        else:
            curr_adding_result = float(line.split(',')[0])
            adding_results.append(curr_adding_result)
    return values, markers, adding_results


def read_data_adding_problem_torch(csv_filename):
    values, markers, adding_results = read_data_adding_problem(csv_filename)
    assert len(values) == len(markers) == len(adding_results)
    num_data = len(values) #always 1000
    seq_len = len(values[0]) #10, 50, 70, 100
    X = torch.Tensor(num_data, seq_len, 2)
    T = torch.Tensor(num_data, 1)
    for k, (curr_values, curr_markers, curr_adding_result) in enumerate(zip(values, markers, adding_results)):
        T[k] = curr_adding_result
        for n, (v, m) in enumerate(zip(curr_values, curr_markers)):
            X[k, n, 0] = v
            X[k, n, 1] = m
    return X, T


def get_batches(X, T, batch_size):
    num_data, max_seq_len, _ = X.shape
    for idx1 in range(0, num_data, batch_size):
        idx2 = min(idx1 + batch_size, num_data)
        yield X[idx1:idx2, :, :], T[idx1:idx2, :]


class SimpleRNNFromBox(nn.Module):
    #many-to-many
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleRNNFromBox, self).__init__()
        self.num_inputs = num_inputs #2
        self.num_hidden = num_hidden #50
        self.num_outputs = num_outputs #1
        self.rnn = nn.RNN(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=1, batch_first=True, bidirectional=False)
        self.out_layer = nn.Linear(in_features=num_hidden, out_features=num_outputs)

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape #max_seq_len = L, num_data = 1000 or 20 if batch
        h0 = torch.zeros(1, num_data, self.num_hidden) # (1, 1000 or 20, 50)
        output, hn = self.rnn(X, h0) # output.shape: num_data x seq_len x num_hidden; (1000 or 20, L, 50)
        last_output = output[:, -1, :] # num_data x num_hidden ;;; many-to-many ==> many-to-one;;; (1000 or 20, 50);;; (batch_size,L,hidden_size) => (batch_size,hidden_size)
        Y = self.out_layer(last_output) # num_data x num_outputs;;; (batch_size,hidden_size) ==>
        return Y


class SimpleLSTMFromBox(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleLSTMFromBox, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=1, batch_first=True, bidirectional=False)
        self.out_layer = nn.Linear(in_features=num_hidden, out_features=num_outputs)

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape
        h0 = torch.zeros(1, num_data, self.num_hidden)
        c0 = torch.zeros(1, num_data, self.num_hidden)
        output, (hn, cn) = self.lstm(X, (h0, c0))
        last_output = output[:, -1, :]
        Y = self.out_layer(last_output) ## Also needed in LSTM?
        return Y

out_times = []

class AlarmworkRNN(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(AlarmworkRNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        ###
        self.W_in1 = nn.Parameter(torch.Tensor(num_hidden, num_inputs).uniform_(-1, 1) * ((6/(num_hidden + num_inputs)) ** 0.5))
        self.b_in1 = nn.Parameter(torch.zeros(num_hidden, 1))
        self.W_rec1 = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-1, 1) * ((6/(num_hidden + num_hidden)) ** 0.5))
        ###
        self.W_in2 = nn.Parameter(torch.Tensor(num_hidden, num_inputs).uniform_(-1, 1) * ((6/(num_hidden + num_inputs)) ** 0.5))
        self.b_in2 = nn.Parameter(torch.zeros(num_hidden, 1))
        self.W_rec2 = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-1, 1) * ((6/(num_hidden + num_hidden)) ** 0.5))
        ###
        self.W_out = nn.Parameter(torch.Tensor(num_outputs, num_hidden).uniform_(-1, 1) * ((6/(num_outputs + num_hidden)) ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(num_outputs, 1))

    def forward0(self, X):
        """
        scalar
        """
        num_data, L, _ = X.shape
        z_in12 = torch.zeros(self.num_hidden, 1)
        z_in1 = torch.zeros(self.num_hidden, 1)
        z_in2 = torch.zeros(self.num_hidden, 1)
        for l in range(L):
            x_small = X[:, l, :].reshape(num_data, 1, self.num_inputs).permute(0, 2, 1)
            a_in1 = torch.matmul(self.W_in1, x_small) + self.b_in1 + torch.matmul(self.W_rec1, z_in12)
            z_in1 = torch.tanh(a_in1)
            if l % 2 == 0:
                a_in2 = torch.matmul(self.W_in2, x_small) + self.b_in2 + torch.matmul(self.W_rec2, z_in2)
                z_in2 = torch.tanh(a_in2)
            z_in12 = z_in1 + z_in2
        out_time = time.time()
        z_out = []
        for b in range(num_data):
            a_out_small = torch.matmul(self.W_out, z_in1[b, :, :]) + self.b_out
            z_out_small = torch.tanh(a_out_small)
            z_out.append(z_out_small)
        z_out = torch.stack(z_out)
        z_out = z_out.reshape(num_data, 1)
        out_times.append(time.time() - out_time)
        return z_out

    def forward(self, X):
        """
        vector
        """
        num_data, L, _ = X.shape
        z_in12 = torch.zeros(self.num_hidden, 1)
        z_in1 = torch.zeros(self.num_hidden, 1)
        z_in2 = torch.zeros(self.num_hidden, 1)
        for l in range(L):
            x_small = X[:, l, :].reshape(num_data, 1, self.num_inputs).permute(0, 2, 1)
            a_in1 = torch.matmul(self.W_in1, x_small) + self.b_in1 + torch.matmul(self.W_rec1, z_in12)
            z_in1 = torch.tanh(a_in1)
            if l % 2 == 0:
                a_in2 = torch.matmul(self.W_in2, x_small) + self.b_in2 + torch.matmul(self.W_rec2, z_in2)
                z_in2 = torch.tanh(a_in2)
            z_in12 = z_in1 + z_in2
        out_time = time.time()
        a_out = torch.matmul(self.W_out, z_in1) + self.b_out
        z_out = torch.tanh(a_out)
        z_out = z_out.reshape(num_data, 1)
        out_times.append(time.time() - out_time)
        return z_out


def adding_problem_evaluate(outputs, gt_outputs):
    assert outputs.shape == gt_outputs.shape
    num_data = outputs.shape[0]
    num_correct = 0
    for i in range(num_data):
        y = outputs[i].item()
        t = gt_outputs[i].item()
        if abs(y - t) < 0.1:
            num_correct += 1
    acc = num_correct*100 / len(outputs)
    return acc


def choose_model(m):
    return m(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)


print('Welcome to the real world!')

table = []
for SEQ_LEN in [10, 50, 70, 100]:
    start_time = time.time()

    X_train, T_train = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_train.csv' % SEQ_LEN)
    X_dev, T_dev = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_dev.csv' % SEQ_LEN)
    X_test, T_test = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_test.csv' % SEQ_LEN)
    print(X_dev.shape)

    mdl_acc = []
    epocs = []
    for mdl in [SimpleRNNFromBox, SimpleLSTMFromBox, AlarmworkRNN]:
        model = choose_model(mdl)
        # model = SimpleRNNFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        # model = SimpleLSTMFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        # model = AlarmworkRNN(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        print(type(model).__name__)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        batch_times = []

        for e in range(50):
            model.eval()
            dev_acc = adding_problem_evaluate(model(X_dev), T_dev)
            print(f'T = {SEQ_LEN}, epoch = {e}, DEV accuracy = {dev_acc}%%')
            if dev_acc > 99.5:
                epocs.append(e)
                break
            model.train()
            for X_batch, T_batch in get_batches(X_train, T_train, batch_size=BATCH_SIZE):
                batch_time = time.time()
                Y_batch = model(X_batch)
                batch_times.append(time.time() - batch_time)
                loss = loss_fn(Y_batch, T_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if e == 49:
                epocs.append(49)

        test_acc = adding_problem_evaluate(model(X_test), T_test)
        mdl_acc.append(test_acc)
        print(f'\nTEST accuracy = {test_acc}')
        if mdl == AlarmworkRNN:
            print(f"Average batch processing time for output layer: {sum(out_times)/len(out_times)} seconds")
        print("All module: --- %s seconds ---" % (round(time.time() - start_time, 2)))
    table.append(f"SEQ_LEN:{SEQ_LEN}")
    table.append("\t\tRNN\t\tLSTM\tAlarm")
    table.append(f"epocs\t{epocs[0]}\t\t{epocs[1]}\t\t{epocs[2]}")
    table.append(f"acc\t\t{mdl_acc[0]}\t{mdl_acc[1]}\t{mdl_acc[2]}")

for t in table:
    print(t)

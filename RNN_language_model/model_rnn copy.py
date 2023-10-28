import torch 
import numpy as np



class NextWordPredictor(torch.nn.Module):
    def __init__(self, rnn_size, vocab_size, embedding_matrix = None ,embedding = False):
        super().__init__()
        self.embed = embedding
        if self.embed:
          self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), padding_idx=0, freeze=True)
          emb_dim = embedding_matrix.shape[1]

          self.rnn = torch.nn.RNN(input_size=emb_dim, hidden_size=rnn_size, num_layers=1, batch_first=True)
        else:
          self.rnn = torch.nn.RNN(input_size=4, hidden_size=rnn_size, num_layers=1, batch_first=True)
        self.fc_logits = torch.nn.Linear(rnn_size, vocab_size)  # Output size should be vocab_size for next word prediction

    def forward(self, inputs):
        inputs =inputs.to(torch.float32)
        if self.embed:
          encoded_inputs = self.embedding(inputs)
          all_states, final_state = self.rnn(encoded_inputs)
        else:
          all_states, final_state = self.rnn(inputs)
        final_state = final_state.squeeze()

        # Apply softmax to get probabilities over the vocabulary
        output_probs = torch.nn.functional.softmax(self.fc_logits(all_states),dim=-1)# dim=-1

        return output_probs


def training_loop(model, num_epochs,train_loader):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        losses = []
        for batch_index, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze(dim=1).long()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}: loss {np.mean(losses)}')
    return model

def evaluate(model, val_loader):
    predictions = []
    labels = []
    with torch.no_grad(): # no backprop for evaluation
        for batch_index, (inputs, targets) in enumerate(val_loader):
            outputs = torch.softmax(model(inputs), 1 ) # apply softmax to get probabilities/logits
    return outputs

def get_perplexity(model, num_epochs,train_loader):
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  for epoch in range(num_epochs):
    losses = []
    for batch_index, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)

            targets = targets.squeeze()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    perplexity = np.exp(sum(losses) / (len(losses)))
    print('Perplexity = ', perplexity)
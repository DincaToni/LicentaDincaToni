import torch
import torch.nn as nn

class QuestionGenerator(nn.Module):
    def __init__(self, dictionarySize, embeddingDim, hiddenSize, numOfLayers):
        super(QuestionGenerator, self).__init__()

        #primul layer realizeaza dictionarul ce cuprinde 'dictionarySize' numar cuvinte. Aceste cuvinte sunt
        #reprezentate de cate un vector de dimensiune 'embeddingDim'. Valorile sunt aleatoare.
        self.dictionar = nn.Embedding(dictionarySize, embeddingDim )
        #testing first layer:
        print("Dictionar: ", self.dictionar.weight)

        #al doilea layer al modelului menit sa modeleze valorile din vectorul reprezentativ fiecarui cuvant astefel alocandu-le un "sens"

        self.lstm = nn.LSTM(embeddingDim, hiddenSize, numOfLayers, batch_first= True)

        #al treilea layer este unul linear
        self.linear = nn.Linear(hiddenSize, dictionarySize)

    def forward(self, inputs):
        #Fiecare cuvant primeste un array de valori
        embededText = self.dictionar(inputs);
        print("testing embedded text: ", embededText)

        lstmOut,_ = self.lstm(embededText);
        print("testing lstmOut: ", lstmOut)
        lstmOut = lstmOut.reshape(-1, lstmOut.shape[2])
        print("testing lstmOut after reshape: ", lstmOut)


        output = self.linear(lstmOut)
        print("testing output: ", output)

        return output
    def generate(self, inputs, maxLen = 20):
        #functia va genera intrebarea pe baza algoritmului beam search
        beamWidth = 4
        topK = 4
        SOS_Token = 0 #Start of sentence
        EOS_Token = 1 #End of sentence
        with torch.no_grad:
            embededText = self.dictionar(inputs)

            lstmOut, (h_n, c_n) = self.lstm(embededText);
            #intrucat lstmOut.size[1] o sa fie mereu 1 deoarece generam cate un cuvant in parte, il concatenam.

            beam = [([],0.0, h_n, c_n)]

            for _ in range (maxLen):
                beamCandidates = []
                for sentence, score, h, c in beam:
                    if(len(sentence) > 0 and sentence[-1] == EOS_Token): #adica <eos>
                        beamCandidates.append((sentence, score))
                    else:
                        #daca nu este <eos> generam urmatorul token (cuvant)
                        if(len(sentence) == 0):
                            inputTensor = torch.tensor([[SOS_Token]]).to(inputs.device)
                            print("test inputTensor: ", inputTensor)
                        else:
                            inputTensor = torch.tensor([sentence[-1]])

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def generate(self, inputs, maxLen = 1, temperature = 1.0):#!!!!!!!!!Don't forget to set bavk to 20 after testing
        #temperature regrleaza cat de mare e diferenta dintre procentaje
        #functia va genera intrebarea pe baza algoritmului beam search
        beamWidth = 4
        topK = 4
        SOS_Token = 0 #Start of sentence
        EOS_Token = 1 #End of sentence
        with torch.no_grad():
            embededText = self.dictionar(inputs)

            lstmOut, (h_n, c_n) = self.lstm(embededText);

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
                            inputTensor = torch.tensor([sentence[-1]]).unsqueeze(0).to(inputs.device)
                            embededText = self.dictionar(inputTensor)
                            #am genrat tokenul initial pe baza dictionarului, urmand sa il modificam utilizand lstm

                            lstmOut, (h_n, c_n) = self.lstm(inputTensor,(h, c))

                            output = self.linear(lstmOut.squeeze(1))

                            probability = F.softmax(output/temperature,1)

                            topKValues, topLIndices = torch.topk(probability, topK, 1)

                            #realizam alegera locala a celor topK probabilitati
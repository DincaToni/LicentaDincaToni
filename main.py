import torch
from QuestionGenerator import QuestionGenerator

#setam dispozitivul pe care lucram (procesor sau gpu)
dispozitiv = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#definim parametrii
dictionarySize = 10
embeddingDim = 4
hiddenSize = 5
numOfLayers = 1

#Initializare model
model = QuestionGenerator(dictionarySize, embeddingDim, hiddenSize, numOfLayers).to(dispozitiv);

#test apelare model
tensor = torch.tensor([[2,4,5],[1,3,6],[3,5,3]]).to(dispozitiv)
model(tensor)

#test generare intrebare
inputTensor = torch.tensor([[2,4,5],[1,3,6],[3,5,3]]).to(dispozitiv)
outputTensor = model.generate(inputTensor)
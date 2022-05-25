from classes import Data
test = Data("varma_PC_properties.txt", "Varma2009_SI.xls", "transporterData.xlsx")

# Testing the two separate methods which don't run upon instantiation 
print(test.processed_descriptors.describe())




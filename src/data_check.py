#Data checker for blossom


import pandas as pd 
import numpy as np





#File directories
confirm_location = "./review_data/features/ASA/confirm.txt"

ASA_training_location = "./review_data/features/ASA/train_ASA.csv"
ASA_testing_location = "./review_data/features/ASA/test_ASA.csv"

blosub_training_location = "./review_data/features/Blocks_substitution_matrix/train_blocksSubstitution.csv"
blosub_testing_location = "./review_data/features/Blocks_substitution_matrix/test_blocksSubstitution.csv"

physico_training_location = "./review_data/features/Physicochemical/train_Phy.csv"
physico_testing_location = "./review_data/features/Physicochemical/test_Phy.csv"

pssm_training_location = "./review_data/features/PSSM/train_PSSM.csv"
pssm_testing_location = "./review_data/features/PSSM/test_PSSM.csv"

solvent_training_location = "./review_data/features/Solvent_exposure/train_solventExposure.csv"
solvent_testing_location = "./review_data/features/Solvent_exposure/test_solventExposure.csv"

#confirm this is the correct location of the data
confirm_file = open(confirm_location, 'r')
confirmdata = confirm_file.read()
print(confirmdata)

#load/prepare the data itself
ASA_training_dataframe = pd.read_csv(ASA_training_location, sep=",")
ASA_testing_dataframe = pd.read_csv(ASA_testing_location, sep=",")

blosub_training_dataframe = pd.read_csv(blosub_training_location, sep=",")
blosub_training_dataframe = blosub_training_dataframe.drop(["hotspot"], axis=1)

blosub_testing_dataframe = pd.read_csv(blosub_testing_location, sep=",")
blosub_testing_dataframe = blosub_testing_dataframe.drop("hotspot", axis=1)

physico_training_dataframe = pd.read_csv(physico_training_location, sep=",")
physico_training_dataframe = physico_training_dataframe.drop(["hotspot"], axis=1)

physico_testing_dataframe = pd.read_csv(physico_testing_location, sep=",")
physico_testing_dataframe = physico_testing_dataframe.drop(["hotspot"], axis=1)

pssm_training_dataframe = pd.read_csv(pssm_training_location, sep=",")
pssm_training_dataframe = pssm_training_dataframe.drop(["hotspot"], axis=1)

pssm_testing_dataframe = pd.read_csv(pssm_testing_location, sep=",")
pssm_testing_dataframe = pssm_testing_dataframe.drop(["hotspot"], axis=1)

solvent_training_dataframe = pd.read_csv(solvent_training_location, sep=",")
solvent_training_dataframe = solvent_training_dataframe.drop(["hotspot"], axis=1)

solvent_testing_dataframe = pd.read_csv(solvent_testing_location, sep=",")
solvent_testing_dataframe = solvent_testing_dataframe.drop(["hotspot"], axis=1)

ASA_training_dataframe = ASA_training_dataframe.replace(to_replace={'hotspot': -1}, value=0)
ASA_testing_dataframe = ASA_testing_dataframe.replace(to_replace={'hotspot': -1}, value=0)

#NOT SURE IF NEEDED?
blosub_training_dataframe = blosub_training_dataframe.replace(to_replace={'hotspot': -1}, value=0)
blosub_testing_dataframe = blosub_testing_dataframe.replace(to_replace={'hotspot': -1}, value=0)
physico_training_dataframe = physico_training_dataframe.replace(to_replace={'hotspot': -1}, value=0)
physico_testing_dataframe = physico_testing_dataframe.replace(to_replace={'hotspot': -1}, value=0)
pssm_training_dataframe = pssm_training_dataframe.replace(to_replace={'hotspot': -1}, value=0)
pssm_testing_dataframe = pssm_testing_dataframe.replace(to_replace={'hotspot': -1}, value=0)
solvent_training_dataframe = solvent_training_dataframe.replace(to_replace={'hotspot': -1}, value=0)
solvent_testing_dataframe = solvent_testing_dataframe.replace(to_replace={'hotspot': -1}, value=0)


print("SOLVENT TRAINING COLUMNS")
print(solvent_training_dataframe.columns)

print("SOLVENT TESTING COLUMNS")
print(solvent_testing_dataframe.columns)

master_training_dataframe = pd.merge(ASA_training_dataframe, blosub_training_dataframe, left_index=True, right_index=True)
master_training_dataframe = pd.merge(master_training_dataframe, physico_training_dataframe, left_index=True, right_index=True)
master_training_dataframe = pd.merge(master_training_dataframe, pssm_training_dataframe, left_index=True, right_index=True)
master_training_dataframe = pd.merge(master_training_dataframe, solvent_training_dataframe, left_index=True, right_index=True)
#master_training_dataframe = master_training_dataframe.reindex(np.random.permutation(master_training_dataframe.index))

master_testing_dataframe = pd.merge(ASA_testing_dataframe, blosub_testing_dataframe, left_index=True, right_index=True)
master_testing_dataframe = pd.merge(master_testing_dataframe, physico_testing_dataframe, left_index=True, right_index=True)
master_testing_dataframe = pd.merge(master_testing_dataframe, pssm_testing_dataframe, left_index=True, right_index=True)
master_testing_dataframe = pd.merge(master_testing_dataframe, solvent_testing_dataframe, left_index=True, right_index=True)
#master_testing_dataframe = master_testing_dataframe.reindex(np.random.permutation(master_testing_dataframe.index))



print("MASTER TRAINING HEAD")
print(master_training_dataframe.head())

print("MASTER TRAINING COLUMNS")
print(master_training_dataframe.columns)

print("MASTER TESTING HEAD")
print(master_testing_dataframe.head())

print("MASTER TESTING COLUMNS")
print(master_testing_dataframe.columns)

number_of_hotspots_training = 0
number_of_nonhotspots_training = 0

for index, row in master_training_dataframe.iterrows():
	if row["hotspot"] == 1:
		number_of_hotspots_training += 1
	elif row["hotspot"] == 0:
		number_of_nonhotspots_training += 1

number_of_hotspots_testing = 0
number_of_nonhotspots_testing = 0

for index, row in master_testing_dataframe.iterrows():
	if row["hotspot"] == 1:
		number_of_hotspots_testing += 1
	elif row["hotspot"] == 0:
		number_of_nonhotspots_testing += 1

print("TRAINING DATASET STATISTICS")
print("NUMBER OF HOTSPOTS:", number_of_hotspots_training)
print("NUMBER OF NON-HOTSPOTS:", number_of_nonhotspots_training)

print("TESTING DATASET STATISTICS")
print("NUMBER OF HOTSPOTS:", number_of_hotspots_testing)
print("NUMBER OF NON-HOTSPOTS:", number_of_nonhotspots_testing)




























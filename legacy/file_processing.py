
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

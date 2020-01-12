#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    anyMissingValuesInTheDataset = dataSet.isnull().values.any()
    return anyMissingValuesInTheDataset
	
#This function is used to check for duplicate records in a given dataSet
def checkForDulicateRecords (dataSet):
    totalRecordsInDataset = len(dataSet.index)
    numberOfUniqueRecordsInDataset = len(dataSet.drop_duplicates().index)
    anyDuplicateRecordsInTheDataset = False if totalRecordsInDataset == numberOfUniqueRecordsInDataset else True 
    print('Total number of records in the dataset: {}\nUnique records in the dataset: {}'.format(totalRecordsInDataset,numberOfUniqueRecordsInDataset))
    return anyDuplicateRecordsInTheDataset


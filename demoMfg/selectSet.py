# User has to choose which file to process, there are 4 of them
def selectSet():
    Set = input("\nEnter a number from 1-4")
    Set = int(Set)
    assert (Set in [1,2,3,4]), "Invalid selection"
    print("\nProcessing set ", Set)
    return Set
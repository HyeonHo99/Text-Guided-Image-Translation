'''
Prompt Editing
[A]
apple2orange : NA
cat2dog : cat -> Siamese cat
cezanne2photo : NA
horse2zebra : horse -> wild horse
summer2winter : summer -> summer mountain
beagle2germanshepherd : beagle -> Beagle dog
goldenretriever2maltese : goldenretriever -> Golden Retriever dog

[B]
apple2orange : orange -> navel orange
cat2dog : dog -> Siberian Husky
cezanne2photo : NA
horse2zebra : zebra -> wild zebra
summer2winter : winter -> winter mountain
beagle2germanshepherd : germanshepherd -> German Shepherd dog
goldenretriever2maltese : maltese -> Maltese
'''

def edit(TEXT_A,TEXT_B):
    for i in range(len(TEXT_A)):
        if TEXT_A[i] == "cat":
            TEXT_A[i] = "Siamese cat"
        elif TEXT_A[i] == "horse":
            TEXT_A[i] = "wild horse"
        elif TEXT_A[i] == "summer":
            TEXT_A[i] = "summer mountain"
        elif TEXT_A[i] == "man":
            TEXT_A[i] = "male face"
        elif TEXT_A[i] == "beagle":
            TEXT_A[i] = "Beagle dog"
        elif TEXT_A[i] == "goldenretriever":
            TEXT_A[i] = "Golden Retriever dog"

    for i in range(len(TEXT_B)):
        if TEXT_B[i] == "orange":
            TEXT_B[i] = "navel orange"
        elif TEXT_B[i] == "dog":
            TEXT_B[i] = "Siberian Husky"
        elif TEXT_B[i] == "zebra":
            TEXT_B[i] = "wild zebra"
        elif TEXT_B[i] == "winter":
            TEXT_B[i] = "winter mountain"
        elif TEXT_B[i] == "woman":
            TEXT_B[i] = "female face"
        elif TEXT_B[i] == "germanshepherd":
            TEXT_B[i] = "German Shepherd dog"
        elif TEXT_B[i] == "maltese":
            TEXT_B[i] = "Maltese dog"

    return TEXT_A,TEXT_B
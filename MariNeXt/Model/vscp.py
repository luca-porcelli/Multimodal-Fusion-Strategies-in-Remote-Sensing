import numpy as np

def VSCP(image, target):
    
    n_augmented = image.shape[0]//2 # Metà del numero di immagini iniziali
    
    image_temp = image[:n_augmented*2,:,:,:].copy() # Utile per conservare i dati originali durante l'augmentation
    target_temp = target[:n_augmented*2,:,:].copy()
    
    image_augmented = []
    target_augmented = []
    for i in range(n_augmented):

        image_temp[i,:,target_temp[i+n_augmented,:,:]!=-1] = image_temp[i+n_augmented,:,target_temp[i+n_augmented,:,:]!=-1] # Copia le parti delle immagini dove i target sono diversi da -1
        image_augmented.append(image_temp[i,:,:].copy()) # Viene aggiunta la i-esima immagine temporanea alla lista 
        
        target_temp[i,target_temp[i+n_augmented,:,:]!=-1] = target_temp[i+n_augmented,target_temp[i+n_augmented,:,:]!=-1]
        target_augmented.append(target_temp[i,:,:].copy())
    
    image_augmented = np.stack(image_augmented)
    target_augmented = np.stack(target_augmented)
    
    return image_augmented, target_augmented
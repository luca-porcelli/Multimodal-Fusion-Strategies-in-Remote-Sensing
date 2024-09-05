import numpy as np

def VSCP(image10, image20, image60, target):
    
    n_augmented = image10.shape[0]//2 # Metà del numero di immagini iniziali
    
    image_temp10 = image10[:n_augmented*2,:,:,:].copy() # Utile per conservare i dati originali durante l'augmentation
    image_temp20 = image20[:n_augmented*2,:,:,:].copy()
    image_temp60 = image60[:n_augmented*2,:,:,:].copy()
    target_temp = target[:n_augmented*2,:,:].copy()
    
    image_augmented10 = []
    image_augmented20 = []
    image_augmented60 = []
    target_augmented = []
    for i in range(n_augmented):

        image_temp10[i,:,target_temp[i+n_augmented,:,:]!=-1] = image_temp10[i+n_augmented,:,target_temp[i+n_augmented,:,:]!=-1] # Copia le parti delle immagini dove i target sono diversi da -1
        image_augmented10.append(image_temp10[i,:,:].copy()) # Viene aggiunta la i-esima immagine temporanea alla lista 
        
        image_temp20[i,:,target_temp[i+n_augmented,:,:]!=-1] = image_temp20[i+n_augmented,:,target_temp[i+n_augmented,:,:]!=-1] # Copia le parti delle immagini dove i target sono diversi da -1
        image_augmented20.append(image_temp20[i,:,:].copy()) # Viene aggiunta la i-esima immagine temporanea alla lista 

        image_temp60[i,:,target_temp[i+n_augmented,:,:]!=-1] = image_temp60[i+n_augmented,:,target_temp[i+n_augmented,:,:]!=-1] # Copia le parti delle immagini dove i target sono diversi da -1
        image_augmented60.append(image_temp60[i,:,:].copy()) # Viene aggiunta la i-esima immagine temporanea alla lista 

        target_temp[i,target_temp[i+n_augmented,:,:]!=-1] = target_temp[i+n_augmented,target_temp[i+n_augmented,:,:]!=-1]
        target_augmented.append(target_temp[i,:,:].copy())
    
    image_augmented10 = np.stack(image_augmented10)
    image_augmented20 = np.stack(image_augmented20)
    image_augmented60 = np.stack(image_augmented60)
    target_augmented = np.stack(target_augmented)
    
    return image_augmented10, image_augmented20, image_augmented60, target_augmented
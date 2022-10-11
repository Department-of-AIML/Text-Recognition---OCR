#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q keras-ocr  #installation')


# In[ ]:


import keras_ocr
import matplotlib.pyplot as plt  #importing libraries


# In[ ]:


pipeline = keras_ocr.pipeline.Pipeline()


# In[ ]:


images = [
    keras_ocr.tools.read(img) for img in ['Image1.png',
                                          'Image2.png'
    ]
]                            #get images


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


len(images)


# In[ ]:


plt.figure(figsize = (10,20))
plt.imshow(images[0])


# In[ ]:


plt.figure(figsize = (10,20))
plt.imshow(images[1])


# In[ ]:


prediction_groups = pipeline.recognize(images)    #get predictions


# In[ ]:


fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, 
                                    predictions=predictions, 
                                    ax=ax)


import numpy as np
import os
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import spacy


def load_dir(dir):
    output_list=[]
    file_list=os.listdir(dir)
    for filename in file_list:
        if 'png' in filename or 'pkl' in filename:
            continue
        path=os.path.join(dir,filename)
        f=open(path,'r')
        tmp=f.readlines()
        f.close()
        tmp=[i.strip() for i in tmp]
        output='\n'.join(tmp)
        output_list.append(output)
    return output_list

def read_file_list(file_list):
    output_list=[]
    for path in file_list:
        if 'png-' in path:
            pass
        elif '.png' in path or '.pkl' in path:
            continue
        f=open(path,'r')
        tmp=f.readlines()
        f.close()
        tmp=[i.strip() for i in tmp]
        output='\n'.join(tmp)
        output_list.append(output)
    return output_list


def load_file(path):
    f=open(path,'r')
    tmp=f.readlines()
    f.close()
    tmp=[i.strip() for i in tmp]
    return tmp

# Compute KL Divergence based on similarity scores
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_similarity(s1,s2,method='spacy',misc=None):
    if method=='spacy':
        doc1 = misc(s1)
        doc2 = misc(s2)
        similarity_score = doc1.similarity(doc2)
    elif method=='transformer':
        # misc: [tokenizer,model]
        import torch
        from scipy.spatial.distance import cosine
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = misc[0]
        model = misc[1]
        def get_embedding(text):
            # Tokenize and encode the text
            input_ids = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids']
            
            # Get the embeddings for the input ids
            with torch.no_grad():
                outputs = model(input_ids)
            
            # Use the mean of the last layer embeddings as the sentence representation
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return embeddings

        # Get their embeddings
        embedding1 = get_embedding(s1)
        embedding2 = get_embedding(s2)
        # Calculate cosine similarity
        similarity_score = 1 - cosine(embedding1, embedding2)
    return similarity_score


def get_divergence(similarity_matrix,i,j,mode='KL'):
    if mode=='KL':
        # Calculate KL Divergence
        p = similarity_matrix[i] / np.sum(similarity_matrix[i])
        q = similarity_matrix[j] / np.sum(similarity_matrix[j])
        divergence = np.sum(p * np.log(p / q))
    return divergence

def visualize(divergence_matrix,save_path,vmax):
    # Create a heatmap with proper memory management
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        norm=matplotlib.colors.Normalize(vmin=0,vmax=vmax)
        im = ax.imshow(divergence_matrix, cmap='viridis', interpolation='nearest',norm=norm)
        plt.colorbar(im, label='Divergence')

        ax.set_title('Divergence Matrix Heatmap')
        ax.set_xticks(range(divergence_matrix.shape[0]))
        ax.set_yticks(range(divergence_matrix.shape[0]))

        # Save the heatmap as an image file (e.g., PNG)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')  # Reduced DPI for memory
    finally:
        # Ensure the figure is always closed to prevent memory leaks
        plt.close(fig)
        plt.close('all')  # Close any remaining figures
